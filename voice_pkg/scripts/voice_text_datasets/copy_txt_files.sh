#!/bin/bash

# 源目录路径
SOURCE_DIR="/home/kuavo/catkin_dt/src/voice_pkg/scripts/voice_text_datasets"

# 目标根目录路径
DESTINATION_ROOT_DIR="/home/kuavo/catkin_dt/src/voice_pkg/scripts/voice_text_datasets_backup"

# 获取当前时间，格式为YYYY-MM-DD_HHMMSS
CURRENT_TIME=$(date "+%Y-%m-%d_%H%M%S")

# 在目标路径下创建以当前时间命名的文件夹
DESTINATION_DIR="${DESTINATION_ROOT_DIR}/${CURRENT_TIME}"
mkdir -p "$DESTINATION_DIR"

# 将所有.txt文件复制到新创建的文件夹中
cp "${SOURCE_DIR}"/*.txt "$DESTINATION_DIR"

echo "文件已复制到$DESTINATION_DIR"
