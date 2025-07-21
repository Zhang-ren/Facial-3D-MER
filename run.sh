#!/bin/bash

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <实验名称> <batch_size> <GPU_ID>"
    echo "示例: $0 exp7_classification 32 0"
    exit 1
fi

# 获取参数
EXP_NAME=$1
BATCH_SIZE=$2
GPU_ID=$3

# 运行训练
echo "开始训练: 实验名称=$EXP_NAME, batch_size=$BATCH_SIZE, GPU=$GPU_ID"
python train_classification.py --exp_name $EXP_NAME --batch_size $BATCH_SIZE --gpu $GPU_ID 