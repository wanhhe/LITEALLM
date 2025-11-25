# eval_kimia.py
import os
import json
import argparse
from typing import List, Dict

from tqdm import tqdm
from jiwer import wer

from kimia_infer.api.kimia import KimiAudio

# 情感分类用
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# =====================
# 通用数据加载函数
# =====================

def load_jsonl(path: str, max_samples: int = -1) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if max_samples > 0 and len(data) >= max_samples:
                break
    return data


def load_parquet(path: str, max_samples: int = -1) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if max_samples > 0:
        df = df.iloc[:max_samples].copy()
    return df


# =====================
# 构造对话 messages
# =====================

def build_asr_messages(audio_path: str, asr_prompt: str) -> List[Dict]:
    """
    ASR 任务：
    - 先给一条文本指令：请将音频内容转换为文字
    - 再给一条 audio message
    """
    messages = [
        {
            "role": "user",
            "message_type": "text",
            "content": asr_prompt,
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        },
    ]
    return messages


def build_emotion_messages(audio_path: str, emotion_prompt: str) -> List[Dict]:
    """
    情感分类任务：
    - 文本指令：请在给定标签中选择一个情绪
    - 再给一条 audio message
    """
    messages = [
        {
            "role": "user",
            "message_type": "text",
            "content": emotion_prompt,
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        },
    ]
    return messages


# =====================
# 情感预测结果解析
# =====================

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


def normalize_emotion_prediction(raw_text: str) -> str:
    """
    把模型生成的文本规整成一个情绪标签。
    假设通过 prompt 已经强约束模型只输出一个英文单词，
    这里再做一层保险：
      - 小写
      - 取首个 token
      - 如果不在 EMOTION_LABELS 中，就原样返回（后面计算 accuracy 时自然会算错）
    """
    if raw_text is None:
        return ""

    txt = raw_text.strip().lower()
    if not txt:
        return ""

    # 只取第一行、第一段
    txt = txt.split("\n")[0].strip()
    # 只取第一个空格前的 token
    token = txt.split()[0]

    if token in EMOTION_LABELS:
        return token

    # 如果没命中，就尝试在整句里找一下
    for lab in EMOTION_LABELS:
        if lab in txt:
            return lab

    # 实在找不到，就返回第一个 token（会被当作错误预测）
    return token


# =====================
# 任务 1: ASR 评估
# =====================

def eval_asr(kimi: KimiAudio, data_path: str, max_samples: int, asr_prompt: str,
             sampling_params: Dict):
    print(f"==> Loading ASR eval data from {data_path}")
    data = load_jsonl(data_path, max_samples=max_samples)
    print(f"Loaded {len(data)} ASR samples.")

    refs: List[str] = []
    hyps: List[str] = []

    print("==> Start ASR inference & WER evaluation ...")
    for sample in tqdm(data, desc="Evaluating ASR"):
        # 取出音频路径和参考文本
        # 这里沿用你原来的数据格式：conversation[1] 是 audio，conversation[2] 是 text
        audio_path = sample["conversation"][1]["content"]
        ref_text = sample["conversation"][2]["content"]

        messages = build_asr_messages(audio_path, asr_prompt=asr_prompt)

        # 只要文字输出即可
        _, hyp_text = kimi.generate(
            messages,
            **sampling_params,
            output_type="text",
        )

        refs.append(ref_text.lower())
        hyps.append(hyp_text.lower())

    corpus_wer = wer(refs, hyps)

    print("========== ASR Evaluation Result ==========")
    print(f"Corpus WER: {corpus_wer:.4f}")
    print("===========================================")

    # 打印前几条 case
    print("\nExample ASR predictions:")
    for i in range(min(5, len(refs))):
        print(f"\n[{i}]")
        print("REF:", refs[i])
        print("HYP:", hyps[i])


# =====================
# 任务 2: Emotion 评估
# =====================

def eval_emotion(
    kimi: KimiAudio,
    data_path: str,
    max_samples: int,
    emotion_prompt: str,
    audio_column: str = "file",
    label_column: str = "emotion",
    sampling_params: Dict = None,
    *,
    audio_from_binary: bool = False,
    tmp_audio_dir: str = "./tmp_kimia_emotion_audio",
):
    """
    如果 audio_from_binary=False:
        直接认为 row[audio_column] 是音频文件路径。
    如果 audio_from_binary=True:
        认为 row[audio_column] 是二进制（bytes/bytearray），
        需要先写成临时文件，再把路径交给 KimiAudio。
    """
    print(f"==> Loading Emotion eval data from {data_path}")
    df = load_parquet(data_path, max_samples=max_samples)
    print(f"Loaded {len(df)} emotion samples.")

    if audio_from_binary:
        os.makedirs(tmp_audio_dir, exist_ok=True)
        print(f"Audio column '{audio_column}' will be treated as binary. "
              f"Temporary wav files will be saved under: {tmp_audio_dir}")

    y_true: List[str] = []
    y_pred: List[str] = []

    print("==> Start Emotion inference & accuracy evaluation ...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Emotion"):
        # -------- 1. 取标签 --------
        gt_label = str(row[label_column]).strip().lower()

        # -------- 2. 准备音频路径 --------
        if not audio_from_binary:
            # 直接当路径用，比如 file 列或 audio 列本来就是 '/path/to.wav'
            audio_path = row[audio_column]
        else:
            audio_raw = row[audio_column]['bytes']

            if audio_raw is None:
                # 没有音频就跳过或当作空预测，这里简单跳过
                continue

            # 二进制情况：假设 audio_raw 是 bytes 或 bytearray，里面已经是完整 wav 文件内容
            if isinstance(audio_raw, (bytes, bytearray)):
                audio_path = os.path.join(tmp_audio_dir, f"sample_{idx}.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_raw)
            else:
                # 如果你发现实际是别的类型，比如 base64 字符串或 numpy 数组，可在这里再加分支
                raise TypeError(
                    f"audio_from_binary=True，但列 '{audio_column}' 类型是 {type(audio_raw)}，"
                    "目前只支持 bytes/bytearray，如有需要请在这里手动转换。"
                )

        # -------- 3. 构造对话并调用模型 --------
        messages = build_emotion_messages(audio_path, emotion_prompt=emotion_prompt)

        _, hyp_text = kimi.generate(
            messages,
            **sampling_params,
            output_type="text",
        )

        pred_label = normalize_emotion_prediction(hyp_text)

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # -------- 4. 计算指标 --------
    acc = accuracy_score(y_true, y_pred)

    print("========== Emotion Evaluation Result ==========")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report (macro metrics):")
    print(classification_report(y_true, y_pred, labels=EMOTION_LABELS))
    print("===============================================")

    print("\nExample Emotion predictions:")
    for i in range(min(5, len(y_true))):
        print(f"\n[{i}]")
        print("GT :", y_true[i])
        print("PRED:", y_pred[i])



# =====================
# main
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="KimiAudio 模型路径（可以是原始或压缩后的目录）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="评估数据路径：ASR 用 jsonl，Emotion 用 parquet",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="asr",
        choices=["asr", "emotion"],
        help="评估任务类型：asr / emotion",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="最多评估多少条样本，-1 表示全部",
    )

    # ASR 任务 prompt
    parser.add_argument(
        "--asr_prompt",
        type=str,
        default="请将音频内容转换为文字。",
        help="ASR 任务喂给模型的文本指令（第一个 user text message）",
    )

    # Emotion 任务 prompt（强约束输出 7 类英文单词）
    parser.add_argument(
        "--emotion_prompt",
        type=str,
        default=(
            "请你判断这段语音中说话人的情绪类别，只能在以下七个英文单词中选择一个："
            "anger, disgust, fear, happiness, neutral, sadness, surprise。"
            "请直接输出其中一个单词作为结果，不要输出其他内容。"
        ),
        help="情感分类任务的文本指令",
    )
    parser.add_argument(
        "--emotion_audio_column",
        type=str,
        default="file",
        help="parquet 中表示音频路径的列名（默认使用 'file' 列）",
    )
    parser.add_argument(
        "--emotion_label_column",
        type=str,
        default="emotion",
        help="parquet 中情绪标签列名（默认 'emotion'）",
    )
    parser.add_argument(
        "--emotion_audio_from_binary",
        action="store_true",
        help="若设置，则认为 parquet 中 emotion_audio_column 存的是二进制音频，需要先写成临时文件再传给模型。",
    )
    parser.add_argument(
        "--tmp_audio_dir",
        type=str,
        default="./tmp_kimia_emotion_audio",
        help="当 emotion_audio_from_binary=True 时，用来存放临时音频文件的目录。",
    )

    # 采样参数
    parser.add_argument("--audio_temperature", type=float, default=0.8)
    parser.add_argument("--audio_top_k", type=int, default=10)
    parser.add_argument("--text_temperature", type=float, default=0.0)
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--audio_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--audio_repetition_window_size", type=int, default=64)
    parser.add_argument("--text_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--text_repetition_window_size", type=int, default=16)

    args = parser.parse_args()

    assert os.path.exists(args.model_path), f"model_path 不存在: {args.model_path}"
    assert os.path.exists(args.data_path), f"data_path 不存在: {args.data_path}"

    print(f"==> Loading KimiAudio model from {args.model_path}")
    kimi = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=False,  # 不需要生成音频
    )

    sampling_params = {
        "audio_temperature": args.audio_temperature,
        "audio_top_k": args.audio_top_k,
        "text_temperature": args.text_temperature,
        "text_top_k": args.text_top_k,
        "audio_repetition_penalty": args.audio_repetition_penalty,
        "audio_repetition_window_size": args.audio_repetition_window_size,
        "text_repetition_penalty": args.text_repetition_penalty,
        "text_repetition_window_size": args.text_repetition_window_size,
    }

    if args.task == "asr":
        eval_asr(
            kimi,
            data_path=args.data_path,
            max_samples=args.max_samples,
            asr_prompt=args.asr_prompt,
            sampling_params=sampling_params,
        )
    else:  # emotion
        eval_emotion(
            kimi,
            data_path=args.data_path,
            max_samples=args.max_samples,
            emotion_prompt=args.emotion_prompt,
            audio_column=args.emotion_audio_column,
            label_column=args.emotion_label_column,
            sampling_params=sampling_params,
            audio_from_binary=args.emotion_audio_from_binary,
            tmp_audio_dir=args.tmp_audio_dir,
        )


if __name__ == "__main__":
    main()


# python eval.py --task emotion --model_path pretrains/Kimi-Audio-7B-Instruct --data_path SAVEE/data/train-00000-of-00001.parquet --emotion_audio_column audio --max_samples 200 --emotion_audio_from_binary
# python eval.py --task asr --model_path pretrains/Kimi-Audio-7B-Instruct --data_path output/data/librispeech/librispeech_with_semantic_codes.jsonl --max_samples 200