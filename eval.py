# eval_kimia.py
import os
import json
import argparse
from typing import List, Dict

from tqdm import tqdm
from jiwer import wer

from kimia_infer.api.kimia import KimiAudio

# For emotion classification
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# =====================
# General Data Loading Functions
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
# Construct Conversation Messages
# =====================

def build_asr_messages(audio_path: str, asr_prompt: str) -> List[Dict]:
    """
    ASR task:
    - First provide a text instruction: convert audio content to text
    - Then provide an audio message
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
    Emotion classification task:
    - Text instruction: Please select an emotion from the given labels
    - Then provide an audio message
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
# Emotion Prediction Result Parsing
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
    Normalize model-generated text into an emotion label.
    Assuming the prompt has already strongly constrained the model to output only one English word,
    add another layer of safety here:
      - Lowercase
      - Take first token
      - If not in EMOTION_LABELS, return as-is (will naturally be counted as wrong in accuracy calculation)
    """
    if raw_text is None:
        return ""

    txt = raw_text.strip().lower()
    if not txt:
        return ""

    # Only take first line, first segment
    txt = txt.split("\n")[0].strip()
    # Only take token before first space
    token = txt.split()[0]

    if token in EMOTION_LABELS:
        return token

    # If not matched, try to find in the entire sentence
    for lab in EMOTION_LABELS:
        if lab in txt:
            return lab

    # If still not found, return first token (will be treated as wrong prediction)
    return token


# =====================
# Task 1: ASR Evaluation
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
        # Extract audio path and reference text
        # Here we follow the original data format: conversation[1] is audio, conversation[2] is text
        audio_path = sample["conversation"][1]["content"]
        ref_text = sample["conversation"][2]["content"]

        messages = build_asr_messages(audio_path, asr_prompt=asr_prompt)

        # Only need text output
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

    # Print first few cases
    print("\nExample ASR predictions:")
    for i in range(min(5, len(refs))):
        print(f"\n[{i}]")
        print("REF:", refs[i])
        print("HYP:", hyps[i])


# =====================
# Task 2: Emotion Evaluation
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
    If audio_from_binary=False:
        Directly treat row[audio_column] as audio file path.
    If audio_from_binary=True:
        Treat row[audio_column] as binary (bytes/bytearray),
        need to write to temp file first, then pass path to KimiAudio.
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
        # -------- 1. Get label --------
        gt_label = str(row[label_column]).strip().lower()

        # -------- 2. Prepare audio path --------
        if not audio_from_binary:
            # Directly use as path, e.g. file column or audio column is already '/path/to.wav'
            audio_path = row[audio_column]
        else:
            audio_raw = row[audio_column]['bytes']

            if audio_raw is None:
                # Skip if no audio or treat as empty prediction, here simply skip
                continue

            # Binary case: assume audio_raw is bytes or bytearray, already contains complete wav file content
            if isinstance(audio_raw, (bytes, bytearray)):
                audio_path = os.path.join(tmp_audio_dir, f"sample_{idx}.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_raw)
            else:
                # If you find it's actually another type, e.g. base64 string or numpy array, add branch here
                raise TypeError(
                    f"audio_from_binary=True, but column '{audio_column}' type is {type(audio_raw)}, "
                    "currently only supports bytes/bytearray, please manually convert here if needed."
                )

        # -------- 3. Construct conversation and call model --------
        messages = build_emotion_messages(audio_path, emotion_prompt=emotion_prompt)

        _, hyp_text = kimi.generate(
            messages,
            **sampling_params,
            output_type="text",
        )

        pred_label = normalize_emotion_prediction(hyp_text)

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # -------- 4. Calculate metrics --------
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
        help="KimiAudio model path (can be original or compressed directory)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Evaluation data path: ASR uses jsonl, Emotion uses parquet",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="asr",
        choices=["asr", "emotion"],
        help="Evaluation task type: asr / emotion",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate, -1 means all",
    )

    # ASR task prompt
    parser.add_argument(
        "--asr_prompt",
        type=str,
        default="Please convert the audio content to text.",
        help="Text instruction for ASR task fed to model (first user text message)",
    )

    # Emotion task prompt (strongly constrains output to 7 English word categories)
    parser.add_argument(
        "--emotion_prompt",
        type=str,
        default=(
            "请你判断这段语音中说话人的情绪类别，只能在以下七个英文单词中选择一个："
            
            "anger, disgust, fear, happiness, neutral, sadness, surprise。"
            "请直接输出其中一个单词作为结果，不要输出其他内容。"
        ),
        help="Text instruction for emotion classification task",
    )
    parser.add_argument(
        "--emotion_audio_column",
        type=str,
        default="file",
        help="Column name in parquet representing audio path (default uses 'file' column)",
    )
    parser.add_argument(
        "--emotion_label_column",
        type=str,
        default="emotion",
        help="Emotion label column name in parquet (default 'emotion')",
    )
    parser.add_argument(
        "--emotion_audio_from_binary",
        action="store_true",
        help="If set, treat emotion_audio_column in parquet as binary audio, "
             "need to write to temp file first before passing to model.",
    )
    parser.add_argument(
        "--tmp_audio_dir",
        type=str,
        default="./tmp_kimia_emotion_audio",
        help="Directory for storing temporary audio files when emotion_audio_from_binary=True.",
    )

    # Sampling parameters
    parser.add_argument("--audio_temperature", type=float, default=0.8)
    parser.add_argument("--audio_top_k", type=int, default=10)
    parser.add_argument("--text_temperature", type=float, default=0.0)
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--audio_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--audio_repetition_window_size", type=int, default=64)
    parser.add_argument("--text_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--text_repetition_window_size", type=int, default=16)

    args = parser.parse_args()

    assert os.path.exists(args.model_path), f"model_path does not exist: {args.model_path}"
    assert os.path.exists(args.data_path), f"data_path does not exist: {args.data_path}"

    print(f"==> Loading KimiAudio model from {args.model_path}")
    kimi = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=False,  # Don't need to generate audio
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