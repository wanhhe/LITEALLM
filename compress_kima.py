from dataclasses import dataclass, field
import json
import logging
import os
import time
import random
from typing import Dict, Optional, List

import librosa
import numpy as np
import pandas as pd
import torch
import transformers
from jiwer import wer
from tqdm import tqdm
from transformers.trainer_pt_utils import LabelSmoother

from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.low_rank_whisper_new import (
    attach_calibration_hooks_to_whisper_encoder,
    apply_low_rank_to_whisper_encoder,
)
from kimia_infer.api.kimia import KimiAudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# ===== Label Sets =====
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

SEC_LABELS = [
    "breath",
    "cough",
    "crying",
    "laugh",
    "screaming",
    "sneeze",
    "yawm",
]


# ========= General Utility Functions =========
def normalize_class_prediction(raw_text: str, valid_labels: List[str]) -> str:
    """
    Normalize text output for classification tasks:
    - Convert to lowercase
    - Only consider the first line and first token
    - Try to match any valid_labels in the entire sentence
    """
    if raw_text is None:
        return ""

    txt = raw_text.strip().lower()
    if not txt:
        return ""

    txt = txt.split("\n")[0].strip()
    token = txt.split()[0]

    if token in valid_labels:
        return token

    for lab in valid_labels:
        if lab in txt:
            return lab

    return token


def build_emotion_messages(audio_path: str, prompt: str):
    return [
        {
            "role": "user",
            "message_type": "text",
            "content": prompt,
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        },
    ]


def build_sec_messages(audio_path: str, prompt: str):
    # Same structure as emotion, only prompt differs
    return build_emotion_messages(audio_path, prompt)


def build_aqa_messages(audio_path: str, question: str, prompt: str):
    """
    AQA: First provide text instruction + question, then audio.
    """
    full_prompt = f"{prompt}\n问题：{question}"
    return [
        {
            "role": "user",
            "message_type": "text",
            "content": full_prompt,
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        },
    ]


def build_ar_messages(
    audio_path: str,
    question: str,
    choices: list,
    ar_prompt: str,
):
    """
    AR task: Multiple choice, but only let the model output **complete option text**.
    choices: List[str], e.g. ["Man", "Woman", "Child", "Robot"]
    """
    # Format choices nicely, but don't use A/B/C/D to avoid model outputting only letters
    choices_str = "\n".join(f"- {c}" for c in choices)

    text = (
        f"{ar_prompt}\n"
        f"问题：{question}\n"
        f"可选答案如下（请从中选择一个完整输出）：\n{choices_str}"
    )

    return [
        {
            "role": "user",
            "message_type": "text",
            "content": text,
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        },
    ]



def ensure_audio_path_from_cell(audio_cell, idx: int, tmp_audio_dir: str) -> str:
    """
    Convert an audio_cell from parquet to an audio file path.

    Currently only supports:
    1) str: Already a path, return directly
    2) dict:
       - bytes: Complete wav bytes -> write to file
    3) bytes/bytearray: Complete wav bytes -> write to file
    To support path / array+sr, can be extended later.
    """
    if isinstance(audio_cell, str):
        return audio_cell

    if isinstance(audio_cell, dict):
        if "bytes" in audio_cell and isinstance(audio_cell["bytes"], (bytes, bytearray)):
            os.makedirs(tmp_audio_dir, exist_ok=True)
            audio_path = os.path.join(tmp_audio_dir, f"sample_{idx}.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_cell["bytes"])
            return audio_path

        raise TypeError(
            f"Unsupported audio dict format at idx={idx}. keys={list(audio_cell.keys())}"
        )

    if isinstance(audio_cell, (bytes, bytearray)):
        os.makedirs(tmp_audio_dir, exist_ok=True)
        audio_path = os.path.join(tmp_audio_dir, f"sample_{idx}.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_cell)
        return audio_path

    raise TypeError(
        f"Unsupported audio cell type at idx={idx}: {type(audio_cell)}"
    )


def ensure_waveform_from_cell(
    audio_cell,
    idx: int,
    tmp_audio_dir: str,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Convert audio_cell to waveform (np.ndarray, shape [T]) for direct input to whisper_model.
    Currently only supports:
    - dict + bytes
    - str path
    - pure bytes
    """
    if isinstance(audio_cell, dict):
        if "bytes" in audio_cell and isinstance(audio_cell["bytes"], (bytes, bytearray)):
            audio_path = ensure_audio_path_from_cell(audio_cell, idx, tmp_audio_dir)
            wav, sr = librosa.load(audio_path, sr=target_sr)
            return wav.astype(np.float32)

        raise TypeError(
            f"Unsupported audio dict format at idx={idx} for waveform. keys={list(audio_cell.keys())}"
        )

    if isinstance(audio_cell, str):
        wav, sr = librosa.load(audio_cell, sr=target_sr)
        return wav.astype(np.float32)

    if isinstance(audio_cell, (bytes, bytearray)):
        audio_path = ensure_audio_path_from_cell(audio_cell, idx, tmp_audio_dir)
        wav, sr = librosa.load(audio_path, sr=target_sr)
        return wav.astype(np.float32)

    raise TypeError(
        f"Unsupported audio cell type for waveform at idx={idx}: {type(audio_cell)}"
    )


# ========= QA / AR Metric Helpers =========
def _normalize_text_for_qa(s: str) -> str:
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text_for_qa(pred).split()
    gold_tokens = _normalize_text_for_qa(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _normalize_ar_answer(s: str) -> str:
    """
    Normalize AR task answer: Prefer first non-empty character (A/B/C/D), otherwise lowercase and strip spaces.
    """
    if s is None:
        return ""
    s = s.strip()
    if not s:
        return ""
    # First check the first letter
    for ch in s:
        if ch.isalpha():
            return ch.lower()
    return s.lower().strip()


# ======================
# Parameter Definitions
# ======================
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="pretrains/Kimi-Audio-7B")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )
    low_rank: bool = False
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    output_dir: str = field(default="output/compress")
    rank_threshold: str = field(default="0.99:0.999")


@dataclass
class DataArguments:
    # ===== Basic: Benchmark data =====
    data_path: str = field(
        default=None, metadata={"help": "Path to the benchmark data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data (for ASR jsonl)."}
    )
    lazy_preprocess: bool = False
    num_calib_samples: int = 1000
    num_test_samples: int = 64

    # Benchmark task type
    task: str = field(
        default="asr",
        metadata={
            "help": "Benchmark task: 'asr', 'emotion', 'sec', 'aqa', 'ar'."
        },
    )

    # ===== Emotion benchmark parameters =====
    emotion_audio_column: str = field(
        default="audio",
        metadata={"help": "Parquet column for audio data in emotion benchmark."},
    )
    emotion_label_column: str = field(
        default="emotion",
        metadata={"help": "Parquet column for emotion label in benchmark."},
    )
    emotion_prompt: str = field(
        default=(
            "请你判断这段语音中说话人的情绪类别，只能在以下七个英文单词中选择一个："
            "anger, disgust, fear, happiness, neutral, sadness, surprise。"
            "请直接输出其中一个单词作为结果，不要输出其他内容。"
        ),
        metadata={"help": "Prompt for emotion classification in benchmark."},
    )
    emotion_audio_from_binary: bool = field(
        default=True,
        metadata={
            "help": "Whether emotion_audio_column stores dict/binary audio that needs temp wav."
        },
    )
    emotion_tmp_audio_dir: str = field(
        default="./tmp_kimia_emotion_bench",
        metadata={"help": "Directory for temporary audio files in emotion benchmark."},
    )

    # ===== Sound Event Classification (SEC) parameters =====
    sec_audio_column: str = field(
        default="audio",
        metadata={"help": "Parquet column for audio data in SEC benchmark."},
    )
    sec_label_column: str = field(
        default="classname",
        metadata={"help": "Parquet column for SEC class name."},
    )
    sec_prompt: str = field(
        default=(
            "请你判断这段音频中主要包含哪一种声音事件，只能在以下七个英文单词中选择一个："
            "breath, cough, crying, laugh, screaming, sneeze, yawm。"
            "请直接输出其中一个单词作为结果，不要输出其他内容。"
        ),
        metadata={"help": "Prompt for sound event classification (SEC)."},
    )
    sec_audio_from_binary: bool = field(
        default=True,
        metadata={
            "help": "Whether sec_audio_column stores dict/binary audio that needs temp wav."
        },
    )
    sec_tmp_audio_dir: str = field(
        default="./tmp_kimia_sec_bench",
        metadata={"help": "Directory for temporary audio files in SEC benchmark."},
    )

    # ===== Audio Question Answering (AQA) parameters =====
    aqa_audio_column: str = field(
        default="audio",
        metadata={"help": "Parquet column for audio data in AQA benchmark."},
    )
    aqa_question_column: str = field(
        default="question",
        metadata={"help": "Parquet column for question text in AQA benchmark."},
    )
    aqa_answer_column: str = field(
        default="answer",
        metadata={"help": "Parquet column for answer text in AQA benchmark."},
    )
    aqa_prompt: str = field(
        default=(
            "请根据给定的音频内容回答下面的问题。"
            "请直接输出答案本身，不要复述问题或加入额外解释。"
        ),
        metadata={"help": "Prompt for Audio Question Answering (AQA)."},
    )
    aqa_audio_from_binary: bool = field(
        default=True,
        metadata={
            "help": "Whether aqa_audio_column stores dict/binary audio that needs temp wav."
        },
    )
    aqa_tmp_audio_dir: str = field(
        default="./tmp_kimia_aqa_bench",
        metadata={"help": "Directory for temporary audio files in AQA benchmark."},
    )

    # ===== Audio Reasoning (AR) parameters =====
    ar_audio_column: str = field(
        default="audio",
        metadata={"help": "Parquet column for audio data in AR benchmark."},
    )
    ar_question_column: str = field(
        default="question",
        metadata={"help": "Parquet column for question text in AR benchmark."},
    )
    ar_choices_column: str = field(
        default="choices",
        metadata={"help": "Parquet column for choices in AR benchmark."},
    )
    ar_answer_column: str = field(
        default="answer",
        metadata={"help": "Parquet column for answer in AR benchmark."},
    )
    ar_prompt: str = field(
        default=(
            "你将听到一段音频，并回答一个多项选择题。"
            "请直接从下面给出的选项中复制并输出**一个完整答案**，不要输出选项字母（如 A/B/C/D），也不要添加任何解释或多余文字。"
        ),
        metadata={"help": "Prompt for Audio Reasoning (multiple-choice)"},
    )
    ar_audio_from_binary: bool = field(
        default=True,
        metadata={
            "help": "Whether ar_audio_column stores dict/binary audio that needs temp wav."
        },
    )
    ar_tmp_audio_dir: str = field(
        default="./tmp_kimia_ar_bench",
        metadata={"help": "Directory for temporary audio files in AR benchmark."},
    )

    # =====⭐ Calibration configuration (can differ from benchmark) =====
    calib_task: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Task used for calibration (collecting activations). "
                "If None, fallback to `task`."
            )
        },
    )
    calib_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Data path for calibration. "
                "If None, fallback to `data_path`."
            )
        },
    )
    calib_audio_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Audio column name for calibration parquet tasks. "
                    "If None, use the default audio column of calib_task."
        },
    )
    calib_tmp_audio_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Temp audio dir for calibration parquet tasks. "
                    "If None, use the default tmp dir of calib_task."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ======================
# Data Loading
# ======================
def make_supervised_data_module(
    whisper_model,
    text_tokenizer,
    data_args: DataArguments,
    max_len: int,
    kimia_token_offset: int,
    *,
    data_path_override: Optional[str] = None,
) -> Dict:
    """
    Construct LazySupervisedDataset for ASR.
    When data_path_override is not None, use that path preferentially.
    """
    dataset_cls = LazySupervisedDataset
    path = data_path_override or data_args.data_path

    rank0_print(f"Loading ASR data from {path} ...")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    random.seed(42)
    random.shuffle(all_data)

    if data_args.eval_ratio > 0:
        split_idx = int(len(all_data) * data_args.eval_ratio)
        eval_data = all_data[:split_idx]
        train_data = all_data[split_idx:]
        assert len(eval_data) > 0, "No evaluation data found"
        assert len(train_data) > 0, "No training data found"
    else:
        eval_data = None
        train_data = all_data

    train_dataset = dataset_cls(
        train_data,
        whisper_model=whisper_model,
        text_tokenizer=text_tokenizer,
        max_len=max_len,
        kimia_token_offset=kimia_token_offset,
        need_speech_token=False,
    )

    if eval_data:
        eval_dataset = dataset_cls(
            eval_data,
            whisper_model=whisper_model,
            text_tokenizer=text_tokenizer,
            max_len=max_len,
            kimia_token_offset=kimia_token_offset,
            need_speech_token=False,
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def extract_ref_and_chats_from_raw(raw_item: dict):
    """
    Extract from raw_item:
      - ref_text: Reference transcription (last assistant text)
      - infer_chats: Input for generate(), remove reference answer text from conversation,
                     and remove audio_tokens from audio (keep only file path)
    """
    conv = raw_item["conversation"]

    ref_text = None
    for msg in reversed(conv):
        if msg.get("role") == "assistant" and msg.get("message_type") == "text":
            ref_text = msg.get("content", "")
            break
    if ref_text is None:
        return None, None

    infer_chats = []
    for msg in conv:
        if msg.get("role") == "assistant" and msg.get("message_type") == "text":
            continue

        new_msg = dict(msg)
        if new_msg.get("role") == "user" and new_msg.get("message_type") == "audio":
            new_msg.pop("audio_tokens", None)

        infer_chats.append(new_msg)

    return ref_text, infer_chats


# ====== Get default audio_column / tmp_dir based on task (for calib) ======
def get_default_audio_column_for_task(task: str, data_args: DataArguments) -> str:
    if task == "emotion":
        return data_args.emotion_audio_column
    if task == "sec":
        return data_args.sec_audio_column
    if task == "aqa":
        return data_args.aqa_audio_column
    if task == "ar":
        return data_args.ar_audio_column
    raise ValueError(f"Parquet-like calib_task '{task}' not supported.")


def get_default_tmp_dir_for_task(task: str, data_args: DataArguments) -> str:
    if task == "emotion":
        return data_args.emotion_tmp_audio_dir
    if task == "sec":
        return data_args.sec_tmp_audio_dir
    if task == "aqa":
        return data_args.aqa_tmp_audio_dir
    if task == "ar":
        return data_args.ar_tmp_audio_dir
    raise ValueError(f"Parquet-like calib_task '{task}' not supported.")


# ======================
# Benchmark Task Functions
# ======================
def benchmark_asr(model, dataset, n: int, desc: str, device: torch.device):
    n = min(n, len(dataset))
    print(f"[{desc}] (ASR) benchmarking on {n} samples (full generate per sample)...")

    total_gen_time = 0.0
    total_wer = 0.0
    used_samples = 0
    total_tokens = 0

    with torch.no_grad():
        warmup = min(2, n)
        for i in range(warmup):
            raw_item = dataset.raw_data[i]
            ref_text, infer_chats = extract_ref_and_chats_from_raw(raw_item)
            if ref_text is None:
                continue

            _wav, _text = model.generate(
                infer_chats,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )

        t0 = time.time()

        for idx in tqdm(range(n), desc=f"Counting Time in {desc}"):
            raw_item = dataset.raw_data[idx]
            ref_text, infer_chats = extract_ref_and_chats_from_raw(raw_item)
            if ref_text is None or len(ref_text.strip()) == 0:
                continue

            t_start = time.time()
            gen_wav, gen_text = model.generate(
                infer_chats,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1

            gen_tokens = model.prompt_manager.text_tokenizer.encode(
                gen_text, bos=False, eos=False
            )
            total_tokens += len(gen_tokens)

            this_wer = wer(ref_text.lower(), gen_text.lower())
            total_wer += this_wer

            if idx < 3:
                print(f"[{desc}] Sample {idx} REF: {ref_text.lower()}")
                print(f"[{desc}] Sample {idx} HYP: {gen_text.lower()}")

        t1 = time.time()
        if used_samples == 0:
            print(f"[{desc}] (ASR) no valid samples for WER, skip.")
            return

        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples
        avg_wer = total_wer / used_samples

        avg_time_per_token = total_gen_time / max(total_tokens, 1)
        tokens_per_second = max(total_tokens, 1) / max(total_gen_time, 1e-8)

        print(f"[{desc}] (ASR) total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample")
        print(f"[{desc}] (ASR) pure generate time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample")
        print(f"[{desc}] (ASR) Average WER over {used_samples} samples: {avg_wer:.4f}")
        print(f"[{desc}] (ASR) total generated tokens: {total_tokens}")
        print(
            f"[{desc}] (ASR) avg {avg_time_per_token*1000:.3f} ms / token "
            f"({tokens_per_second:.1f} tokens/s)"
        )

        # Encoder speed measurement
        used_samples = 0
        total_gen_time = 0.0
        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Encoding Time in {desc} (ASR)"):
            sample = dataset[idx]
            wav_np = sample["whisper_input_feature"][0]
            wav_tensor = torch.tensor(
                wav_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            t_start = time.time()
            _ = model.prompt_manager.whisper_model(wav_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1
        t1 = time.time()

        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        print(
            f"[{desc}] (ASR) Encoder total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (ASR) Encoder pure forward time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )


def benchmark_emotion(
    model,
    df: pd.DataFrame,
    n: int,
    desc: str,
    device: torch.device,
    data_args: DataArguments,
):
    assert isinstance(df, pd.DataFrame)
    n = min(n, len(df))
    print(f"[{desc}] (Emotion) benchmarking on {n} samples ...")

    col_audio = data_args.emotion_audio_column
    col_label = data_args.emotion_label_column
    tmp_dir = data_args.emotion_tmp_audio_dir
    prompt = data_args.emotion_prompt
    audio_from_binary = data_args.emotion_audio_from_binary

    total_gen_time = 0.0
    used_samples = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        warmup = min(2, n)
        for i in range(warmup):
            row = df.iloc[i]
            audio_cell = row[col_audio]
            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, i, tmp_dir)
            else:
                audio_path = audio_cell
            messages = build_emotion_messages(audio_path, prompt)
            print(messages)
            _ = model.generate(
                messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )

        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Counting Time in {desc} (Emotion)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            gt_label = str(row[col_label]).strip().lower()

            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, idx, tmp_dir)
            else:
                audio_path = audio_cell

            messages = build_emotion_messages(audio_path, prompt)

            t_start = time.time()
            gen_wav, gen_text = model.generate(
                messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1

            gen_tokens = model.prompt_manager.text_tokenizer.encode(
                gen_text, bos=False, eos=False
            )
            total_tokens += len(gen_tokens)

            pred_label = normalize_class_prediction(gen_text, EMOTION_LABELS)

            if pred_label == gt_label:
                correct += 1

            if idx < 3:
                print(f"[{desc}] Sample {idx} GT   : {gt_label}")
                print(f"[{desc}] Sample {idx} PRED : {pred_label}")
                print(f"[{desc}] Sample {idx} RAW  : {gen_text}")

        t1 = time.time()
        if used_samples == 0:
            print(f"[{desc}] (Emotion) no valid samples, skip.")
            return

        acc = correct / used_samples
        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        avg_time_per_token = total_gen_time / max(total_tokens, 1)
        tokens_per_second = max(total_tokens, 1) / max(total_gen_time, 1e-8)

        print(
            f"[{desc}] (Emotion) total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (Emotion) pure generate time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )
        print(f"[{desc}] (Emotion) Accuracy over {used_samples} samples: {acc:.4f}")
        print(
            f"[{desc}] (Emotion) total generated tokens: {total_tokens}"
        )
        print(
            f"[{desc}] (Emotion) avg {avg_time_per_token*1000:.3f} ms / token "
            f"({tokens_per_second:.1f} tokens/s)"
        )

        # Encoder speed measurement
        used_samples = 0
        total_gen_time = 0.0
        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Encoding Time in {desc} (Emotion)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            wav_np = ensure_waveform_from_cell(audio_cell, idx, tmp_dir, target_sr=16000)
            wav_tensor = torch.tensor(
                wav_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            t_start = time.time()
            _ = model.prompt_manager.whisper_model(wav_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1
        t1 = time.time()

        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        print(
            f"[{desc}] (Emotion) Encoder total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (Emotion) Encoder pure forward time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )


def benchmark_sec(
    model,
    df: pd.DataFrame,
    n: int,
    desc: str,
    device: torch.device,
    data_args: DataArguments,
):
    assert isinstance(df, pd.DataFrame)
    n = min(n, len(df))
    print(f"[{desc}] (SEC) benchmarking on {n} samples ...")

    col_audio = data_args.sec_audio_column
    col_label = data_args.sec_label_column
    tmp_dir = data_args.sec_tmp_audio_dir
    prompt = data_args.sec_prompt
    audio_from_binary = data_args.sec_audio_from_binary

    total_gen_time = 0.0
    used_samples = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        warmup = min(2, n)
        for i in range(warmup):
            row = df.iloc[i]
            audio_cell = row[col_audio]
            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, i, tmp_dir)
            else:
                audio_path = audio_cell
            messages = build_sec_messages(audio_path, prompt)
            _ = model.generate(
                messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )

        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Counting Time in {desc} (SEC)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            gt_label = str(row[col_label]).strip().lower()

            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, idx, tmp_dir)
            else:
                audio_path = audio_cell

            messages = build_sec_messages(audio_path, prompt)

            t_start = time.time()
            gen_wav, gen_text = model.generate(
                messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1

            gen_tokens = model.prompt_manager.text_tokenizer.encode(
                gen_text, bos=False, eos=False
            )
            total_tokens += len(gen_tokens)

            pred_label = normalize_class_prediction(gen_text, SEC_LABELS)

            if pred_label == gt_label:
                correct += 1

            if idx < 3:
                print(f"[{desc}] Sample {idx} GT   : {gt_label}")
                print(f"[{desc}] Sample {idx} PRED : {pred_label}")
                print(f"[{desc}] Sample {idx} RAW  : {gen_text}")

        t1 = time.time()
        if used_samples == 0:
            print(f"[{desc}] (SEC) no valid samples, skip.")
            return

        acc = correct / used_samples
        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        avg_time_per_token = total_gen_time / max(total_tokens, 1)
        tokens_per_second = max(total_tokens, 1) / max(total_gen_time, 1e-8)

        print(
            f"[{desc}] (SEC) total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (SEC) pure generate time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )
        print(f"[{desc}] (SEC) Accuracy over {used_samples} samples: {acc:.4f}")
        print(
            f"[{desc}] (SEC) total generated tokens: {total_tokens}"
        )
        print(
            f"[{desc}] (SEC) avg {avg_time_per_token*1000:.3f} ms / token "
            f"({tokens_per_second:.1f} tokens/s)"
        )

        # Encoder speed measurement
        used_samples = 0
        total_gen_time = 0.0
        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Encoding Time in {desc} (SEC)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            wav_np = ensure_waveform_from_cell(audio_cell, idx, tmp_dir, target_sr=16000)
            wav_tensor = torch.tensor(
                wav_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            t_start = time.time()
            _ = model.prompt_manager.whisper_model(wav_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1
        t1 = time.time()

        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        print(
            f"[{desc}] (SEC) Encoder total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (SEC) Encoder pure forward time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )


def benchmark_aqa(
    model,
    df: pd.DataFrame,
    n: int,
    desc: str,
    device: torch.device,
    data_args: DataArguments,
):
    assert isinstance(df, pd.DataFrame)
    n = min(n, len(df))
    print(f"[{desc}] (AQA) benchmarking on {n} samples ...")

    col_audio = data_args.aqa_audio_column
    col_q = data_args.aqa_question_column
    col_a = data_args.aqa_answer_column
    tmp_dir = data_args.aqa_tmp_audio_dir
    prompt = data_args.aqa_prompt
    audio_from_binary = data_args.aqa_audio_from_binary

    total_gen_time = 0.0
    used_samples = 0
    total_tokens = 0
    em_correct = 0
    f1_sum = 0.0

    with torch.no_grad():
        warmup = min(2, n)
        for i in range(warmup):
            row = df.iloc[i]
            audio_cell = row[col_audio]
            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, i, tmp_dir)
            else:
                audio_path = audio_cell
            question = str(row[col_q])
            messages = build_aqa_messages(audio_path, question, prompt)
            _ = model.generate(
                messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )

        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Counting Time in {desc} (AQA)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            question = str(row[col_q])
            gold_answer = str(row[col_a])

            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, idx, tmp_dir)
            else:
                audio_path = audio_cell

            messages = build_aqa_messages(audio_path, question, prompt)

            t_start = time.time()
            gen_wav, gen_text = model.generate(
                messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1

            gen_tokens = model.prompt_manager.text_tokenizer.encode(
                gen_text, bos=False, eos=False
            )
            total_tokens += len(gen_tokens)

            pred = gen_text
            if _normalize_text_for_qa(pred) == _normalize_text_for_qa(gold_answer):
                em_correct += 1
            f1_sum += _f1_score(pred, gold_answer)

            if idx < 3:
                print(f"[{desc}] Sample {idx} Q   : {question}")
                print(f"[{desc}] Sample {idx} GOLD: {gold_answer}")
                print(f"[{desc}] Sample {idx} PRED: {gen_text}")

        t1 = time.time()
        if used_samples == 0:
            print(f"[{desc}] (AQA) no valid samples, skip.")
            return

        em = em_correct / used_samples
        f1 = f1_sum / used_samples
        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        avg_time_per_token = total_gen_time / max(total_tokens, 1)
        tokens_per_second = max(total_tokens, 1) / max(total_gen_time, 1e-8)

        print(
            f"[{desc}] (AQA) total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (AQA) pure generate time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )
        print(f"[{desc}] (AQA) EM over {used_samples} samples: {em:.4f}")
        print(f"[{desc}] (AQA) F1 over {used_samples} samples: {f1:.4f}")
        print(
            f"[{desc}] (AQA) total generated tokens: {total_tokens}"
        )
        print(
            f"[{desc}] (AQA) avg {avg_time_per_token*1000:.3f} ms / token "
            f"({tokens_per_second:.1f} tokens/s)"
        )

        # Encoder speed measurement
        used_samples = 0
        total_gen_time = 0.0
        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Encoding Time in {desc} (AQA)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            wav_np = ensure_waveform_from_cell(audio_cell, idx, tmp_dir, target_sr=16000)
            wav_tensor = torch.tensor(
                wav_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            t_start = time.time()
            _ = model.prompt_manager.whisper_model(wav_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1
        t1 = time.time()

        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        print(
            f"[{desc}] (AQA) Encoder total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (AQA) Encoder pure forward time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )

def _norm_str(s: str) -> str:
    if s is None:
        return ""
    # Lowercase + strip whitespace + merge extra spaces + remove leading/trailing quotes
    s = str(s).strip().strip("\"'").lower()
    s = " ".join(s.split())
    return s

def get_ar_gold_choice(answer_field, choices: list) -> str:
    """
    Map answer field in dataset to a choice text:
    - If answer is number/numeric string: get choices[int(answer)] by index
    - If answer is string: find the best matching one in choices
    """
    if isinstance(answer_field, (int, np.integer)) or (
        isinstance(answer_field, str) and answer_field.isdigit()
    ):
        idx = int(answer_field)
        if 0 <= idx < len(choices):
            return choices[idx]
        # Out of range, return original directly
        return str(answer_field)

    # Otherwise treat as string matching
    ans_norm = _norm_str(answer_field)
    norm_choices = [_norm_str(c) for c in choices]

    # Exact match
    for c, nc in zip(choices, norm_choices):
        if ans_norm == nc:
            return c

    # Substring match
    for c, nc in zip(choices, norm_choices):
        if nc in ans_norm or ans_norm in nc:
            return c

    # If all else fails, return original answer
    return str(answer_field)


def match_ar_pred_to_choice(pred_text: str, choices: list) -> str:
    """
    Find the most likely choice from model-generated text:
    - Only take the first line
    - Try to match to some option text
    """
    if pred_text is None:
        return ""

    # First take the first line to prevent multi-line babble
    line = pred_text.strip().split("\n")[0].strip().strip("\"'")
    pred_norm = _norm_str(line)
    norm_choices = [_norm_str(c) for c in choices]

    # 1) Exact match
    for c, nc in zip(choices, norm_choices):
        if pred_norm == nc:
            return c

    # 2) Substring relationship (option in answer / answer is substring of option)
    for c, nc in zip(choices, norm_choices):
        if nc in pred_norm or pred_norm in nc:
            return c

    # 3) Further find first matching option fragment from answer
    for c in choices:
        if _norm_str(c) in pred_norm:
            return c

    # 4) If still not found, return normalized sentence; accuracy will be wrong, that's fine
    return line

def benchmark_ar(
    model,
    df: pd.DataFrame,
    num_samples: int,
    desc: str,
    device: torch.device,
    data_args: DataArguments,
):
    """
    AR: AudioReason multiple choice task
    data_args needs to have:
      - ar_audio_column: Column name for audio (usually 'audio')
      - ar_question_column: Column name for question (usually 'question')
      - ar_choices_column: Column name for choices (usually 'choices', list or JSON string)
      - ar_answer_column: Column name for answer (usually 'answer')
      - ar_prompt: Prompt (AR_PROMPT_DEFAULT defined above)
      - ar_audio_from_binary: Whether it's dict/bytes, needs to write temp file
      - tmp_audio_dir: Temporary audio directory
    """
    assert isinstance(df, pd.DataFrame), "AR task expects dataset to be pandas.DataFrame"
    n = min(num_samples, len(df))
    print(f"[{desc}] (AR) benchmarking on {n} samples (full generate per sample)...")

    col_audio = data_args.ar_audio_column
    col_question = data_args.ar_question_column
    col_choices = data_args.ar_choices_column
    col_answer = data_args.ar_answer_column
    tmp_dir = data_args.ar_tmp_audio_dir
    audio_from_binary = data_args.ar_audio_from_binary
    ar_prompt = data_args.ar_prompt

    total_gen_time = 0.0
    used_samples = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        # Warmup
        warmup = min(2, n)
        for i in range(warmup):
            row = df.iloc[i]
            audio_cell = row[col_audio]

            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, i, tmp_dir)
            else:
                audio_path = audio_cell

            # Parse choices
            choices_raw = row[col_choices]
            if isinstance(choices_raw, str):
                try:
                    choices = json.loads(choices_raw)
                except Exception:
                    # You can also manually split based on actual format, e.g. using '||'
                    choices = [choices_raw]
            else:
                choices = list(choices_raw)

            question = str(row[col_question])

            messages = build_ar_messages(
                audio_path=audio_path,
                question=question,
                choices=choices,
                ar_prompt=ar_prompt,
            )

            _ = model.generate(
                chats=messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )

        # Official timing + accuracy
        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Counting Time in {desc} (AR)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]

            if audio_from_binary:
                audio_path = ensure_audio_path_from_cell(audio_cell, idx, tmp_dir)
            else:
                audio_path = audio_cell

            choices_raw = row[col_choices]
            if isinstance(choices_raw, str):
                try:
                    choices = json.loads(choices_raw)
                except Exception:
                    choices = [choices_raw]
            else:
                choices = list(choices_raw)

            question = str(row[col_question])
            gold_answer_field = row[col_answer]
            gold_choice = get_ar_gold_choice(gold_answer_field, choices)

            messages = build_ar_messages(
                audio_path=audio_path,
                question=question,
                choices=choices,
                ar_prompt=ar_prompt,
            )

            t_start = time.time()
            gen_wav, gen_text = model.generate(
                chats=messages,
                output_type="text",
                audio_temperature=0.8,
                audio_top_k=10,
                text_temperature=0.0,
                text_top_k=5,
            )
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1

            gen_tokens = model.prompt_manager.text_tokenizer.encode(
                gen_text, bos=False, eos=False
            )
            total_tokens += len(gen_tokens)

            pred_choice = match_ar_pred_to_choice(gen_text, choices)

            if _norm_str(pred_choice) == _norm_str(gold_choice):
                correct += 1

            if idx < 3:
                print(f"[{desc}] Sample {idx} Q    : {question}")
                print(f"[{desc}] Sample {idx} GOLD : {gold_choice}")
                print(f"[{desc}] Sample {idx} PRED : {pred_choice}")
                print(f"[{desc}] Sample {idx} RAW  : {gen_text}")

        t1 = time.time()
        if used_samples == 0:
            print(f"[{desc}] (AR) no valid samples, skip.")
            return

        acc = correct / used_samples
        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples
        avg_time_per_token = total_gen_time / max(total_tokens, 1)
        tokens_per_second = max(total_tokens, 1) / max(total_gen_time, 1e-8)

        print(f"[{desc}] (AR) total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample")
        print(f"[{desc}] (AR) pure generate time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample")
        print(f"[{desc}] (AR) Accuracy over {used_samples} samples: {acc:.4f}")
        print(f"[{desc}] (AR) total generated tokens: {total_tokens}")
        print(f"[{desc}] (AR) avg {avg_time_per_token*1000:.3f} ms / token "
              f"({tokens_per_second:.1f} tokens/s)")

        # Encoder speed measurement
        used_samples = 0
        total_gen_time = 0.0
        t0 = time.time()
        for idx in tqdm(range(n), desc=f"Encoding Time in {desc} (AQA)"):
            row = df.iloc[idx]
            audio_cell = row[col_audio]
            wav_np = ensure_waveform_from_cell(audio_cell, idx, tmp_dir, target_sr=16000)
            wav_tensor = torch.tensor(
                wav_np, dtype=torch.float32, device=device
            ).unsqueeze(0)

            t_start = time.time()
            _ = model.prompt_manager.whisper_model(wav_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            total_gen_time += (t_end - t_start)
            used_samples += 1
        t1 = time.time()

        wall_clock = t1 - t0
        avg_wall = wall_clock / used_samples
        avg_gen = total_gen_time / used_samples

        print(
            f"[{desc}] (AR) Encoder total wall time {wall_clock:.2f}s, wall avg {avg_wall:.3f}s / sample"
        )
        print(
            f"[{desc}] (AR) Encoder pure forward time sum {total_gen_time:.2f}s, avg {avg_gen:.3f}s / sample"
        )


# ======================
# Unified Entry: benchmark_model_single
# ======================
def benchmark_model_single(
    model,
    dataset,
    num_samples: int,
    desc: str,
    device: torch.device,
    *,
    task: str = "asr",
    data_args: Optional[DataArguments] = None,
):
    if task == "asr":
        return benchmark_asr(model, dataset, num_samples, desc, device)
    if task == "emotion":
        return benchmark_emotion(model, dataset, num_samples, desc, device, data_args)
    if task == "sec":
        return benchmark_sec(model, dataset, num_samples, desc, device, data_args)
    if task == "aqa":
        return benchmark_aqa(model, dataset, num_samples, desc, device, data_args)
    if task == "ar":
        return benchmark_ar(model, dataset, num_samples, desc, device, data_args)
    raise ValueError(f"Unknown task for benchmark: {task}")


# ======================
# Main Entry
# ======================
def run():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")
    print(f"Loading model from {model_args.model_path}...")
    model = KimiAudio(model_args.model_path, load_detokenizer=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_tokenizer = model.prompt_manager.text_tokenizer
    whisper_model = model.prompt_manager.whisper_model

    # ========= 1) Benchmark data loading =========
    if data_args.task == "asr":
        bench_dm = make_supervised_data_module(
            whisper_model=whisper_model,
            text_tokenizer=text_tokenizer,
            data_args=data_args,
            max_len=model_args.model_max_length,
            kimia_token_offset=model.kimia_token_offset,
        )
        bench_dataset = bench_dm["train_dataset"]
        print(f"Loaded {len(bench_dataset)} ASR benchmark samples from {data_args.data_path}")
    else:
        import glob
        if os.path.isdir(data_args.data_path):
            files = glob.glob(os.path.join(data_args.data_path, "*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found in folder: {data_args.data_path}")
            dfs = [pd.read_parquet(f) for f in files]
            bench_dataset = pd.concat(dfs, ignore_index=True)
        else:
            bench_dataset = pd.read_parquet(data_args.data_path)
        bench_dataset = bench_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Loaded {len(bench_dataset)} '{data_args.task}' benchmark samples from {data_args.data_path}")

    # ====== Benchmark before compression ======
    benchmark_model_single(
        model,
        bench_dataset,
        data_args.num_test_samples,
        desc=f"Before low-rank ({data_args.task})",
        device=device,
        task=data_args.task,
        data_args=data_args,
    )

    # ========= 2) Low-rank compression & calibration =========
    if model_args.low_rank:
        whisper_encoder = whisper_model.speech_encoder
        attach_calibration_hooks_to_whisper_encoder(whisper_encoder)

        whisper_encoder.is_calibrating = True
        whisper_model.eval()

        device = next(whisper_model.parameters()).device

        # Select "calibration" task and data_path (can differ from benchmark)
        calib_task = data_args.calib_task
        calib_data_path = data_args.calib_data_path

        print(f"[Calib] Using task = {calib_task}, data_path = {calib_data_path}")

        if calib_task == "asr":
            calib_dm = make_supervised_data_module(
                whisper_model=whisper_model,
                text_tokenizer=text_tokenizer,
                data_args=data_args,
                max_len=model_args.model_max_length,
                kimia_token_offset=model.kimia_token_offset,
                data_path_override=calib_data_path,
            )
            calib_dataset = calib_dm["train_dataset"]

            num_calib = min(data_args.num_calib_samples, len(calib_dataset))
            indices = np.random.permutation(len(calib_dataset))[:num_calib]
            for i in tqdm(indices, desc="Collecting calibration data (ASR)"):
                sample = calib_dataset[i]
                wav_np = sample["whisper_input_feature"][0]
                wav_tensor = torch.tensor(
                    wav_np, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    _ = whisper_model(wav_tensor)

        # else:
        #     # Parquet-like tasks: only need audio
        #     import glob
        #     if os.path.isdir(calib_data_path):
        #         files = glob.glob(os.path.join(calib_data_path, "*.parquet"))
        #         if not files:
        #             raise FileNotFoundError(f"No parquet files found in folder: {calib_data_path}")
        #         dfs = [pd.read_parquet(f) for f in files]
        #         calib_df = pd.concat(dfs, ignore_index=True)
            # else:
            #     calib_df = pd.read_parquet(calib_data_path)

            # calib_df = calib_df.sample(frac=1, random_state=42).reset_index(drop=True)
            # audio_col = data_args.calib_audio_column or get_default_audio_column_for_task(
            #     calib_task, data_args
            # )
            # tmp_dir = data_args.calib_tmp_audio_dir or get_default_tmp_dir_for_task(
            #     calib_task, data_args
            # )

            # num_calib = min(data_args.num_calib_samples, len(calib_df))
            # for i in tqdm(range(num_calib), desc=f"Collecting calibration data ({calib_task})"):
            #     audio_cell = calib_df.iloc[i][audio_col]
            #     wav_np = ensure_waveform_from_cell(audio_cell, i, tmp_dir, target_sr=16000)
            #     wav_tensor = torch.tensor(
            #         wav_np, dtype=torch.float32, device=device
            #     ).unsqueeze(0)
            #     with torch.no_grad():
            #         _ = whisper_model(wav_tensor)
        else:
            # Parquet-like tasks: only need audio
            import glob
        
            # Check if using same dataset & task as benchmark
            same_as_bench = (
                calib_data_path == data_args.data_path
                and calib_task == data_args.task
            )
        
            if same_as_bench:
                # Directly reuse the already loaded bench_dataset (already shuffled)
                calib_df = bench_dataset
            else:
                if os.path.isdir(calib_data_path):
                    files = glob.glob(os.path.join(calib_data_path, "*.parquet"))
                    if not files:
                        raise FileNotFoundError(f"No parquet files found in folder: {calib_data_path}")
                    dfs = [pd.read_parquet(f) for f in files]
                    calib_df = pd.concat(dfs, ignore_index=True)
                else:
                    calib_df = pd.read_parquet(calib_data_path)
                # Independent dataset for calibration, shuffle separately
                calib_df = calib_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
            audio_col = data_args.calib_audio_column or get_default_audio_column_for_task(
                calib_task, data_args
            )
            tmp_dir = data_args.calib_tmp_audio_dir or get_default_tmp_dir_for_task(
                calib_task, data_args
            )
        
            total = len(calib_df)
            num_calib = min(data_args.num_calib_samples, total)
        
            if same_as_bench:
                # Avoid the first num_test_samples rows used by benchmark
                n_bench = min(data_args.num_test_samples, total)
                start = n_bench
                end = min(n_bench + num_calib, total)
                # First use [n_bench, end) portion
                base_indices = list(range(start, end))
        
                if len(base_indices) < num_calib:
                    # If not enough, randomly supplement from [0, n_bench)
                    rest_needed = num_calib - len(base_indices)
                    head = np.random.choice(np.arange(0, n_bench), size=rest_needed, replace=False)
                    indices = base_indices + head.tolist()
                else:
                    indices = base_indices
            else:
                # Different dataset, directly use first num_calib (calib_df already shuffled once)
                indices = list(range(num_calib))
        
            for i in tqdm(indices, desc=f"Collecting calibration data ({calib_task})"):
                audio_cell = calib_df.iloc[int(i)][audio_col]
                wav_np = ensure_waveform_from_cell(audio_cell, int(i), tmp_dir, target_sr=16000)
                wav_tensor = torch.tensor(
                    wav_np, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    _ = whisper_model(wav_tensor)


        whisper_encoder.is_calibrating = False

        _, stats = apply_low_rank_to_whisper_encoder(
            whisper_encoder,
            rank_threshold=model_args.rank_threshold,
        )
        # print(stats)

        # ===== Benchmark after compression =====
        benchmark_model_single(
            model,
            bench_dataset,
            data_args.num_test_samples,
            desc=f"After low-rank ({data_args.task})",
            device=device,
            task=data_args.task,
            data_args=data_args,
        )


if __name__ == "__main__":
    run()
