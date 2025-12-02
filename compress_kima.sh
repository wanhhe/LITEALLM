#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(/root/miniconda3/envs/kimi_audio/bin/python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

python compress_kima.py \
  --model_path pretrains/Kimi-Audio-7B-Instruct \
  --task sec \
  --data_path nonspeech7k \
  --output_dir output/compress \
  --model_max_length 1024 \
  --rank_threshold "0.99:0.99" \
  --num_test_samples 100 \
  --low_rank True \
  --calib_task sec \
  --calib_data_path nonspeech7k \
  --num_calib_samples 200

# python compress_kima.py \
#   --model_path pretrains/Kimi-Audio-7B-Instruct \
#   --task emotion \
#   --data_path SAVEE \
#   --output_dir output/compress \
#   --model_max_length 1024 \
#   --rank_threshold "0.985:0.985" \
#   --num_test_samples 100 \
#   --low_rank True \
#   --calib_task asr \
#   --calib_data_path output/data/librispeech/librispeech.jsonl \
#   --num_calib_samples 100

# python compress_kima.py \
#   --model_path pretrains/Kimi-Audio-7B-Instruct \
#   --task ar \
#   --data_path mmau \
#   --output_dir output/compress \
#   --model_max_length 1024 \
#   --rank_threshold "0.985:0.985" \
#   --num_test_samples 100 \
#   --low_rank True \
#   --calib_task asr \
#   --calib_data_path output/data/librispeech/librispeech.jsonl \
#   --num_calib_samples 100

# python compress_kima.py \
#   --model_path pretrains/Kimi-Audio-7B-Instruct \
#   --task aqa \
#   --data_path clothoaqa \
#   --output_dir output/compress \
#   --model_max_length 1024 \
#   --rank_threshold "0.99:0.99" \
#   --num_test_samples 100 \
#   --low_rank True \
#   --calib_task asr \
#   --calib_data_path output/data/librispeech/librispeech.jsonl \
#   --num_calib_samples 100