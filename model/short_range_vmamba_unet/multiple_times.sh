#!/bin/bash

# 启用set -e，确保脚本遇到错误时退出
set -e

# 设置训练脚本路径
TRAIN_SCRIPT="/home/czh/vmamba-unet/model/short_range_vmamba_unet/train_vmamaba_pre_entire.py"

# 定义一个数组包含所有的尺寸
SIZES=("384,384")

# 对每个尺寸执行5次训练
for size in "${SIZES[@]}"
do
  echo "开始对尺寸 $size 进行第 1 次训练..."
  
  # 导出当前尺寸到环境变量 IMAGE_SIZE
  export IMAGE_SIZE=$size
    # 使用python命令执行训练脚本
  python $TRAIN_SCRIPT
  
  echo "尺寸 $size 的第 1 次训练完成。"

done

echo "所有训练已完成。"
