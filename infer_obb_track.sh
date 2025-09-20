#!/bin/bash
# chmod +x infer_obb_track.sh
set -e  # 遇到错误立即退出

# 日志目录（自动创建）
LOGDIR="./logs_obb_track"
OUTPUT_DIR="./results"
mkdir -p "$LOGDIR"
mkdir -p "$OUTPUT_DIR"

# 日志文件名（自动加日期）
LOGFILE="$LOGDIR/run_$(date +'%Y-%m-%d_%H-%M-%S').log"

echo "======================================" > "$LOGFILE"
echo " Run started at $(date)" >> "$LOGFILE"
echo "======================================" >> "$LOGFILE"

# # 只写入日志文件，不在终端显示
# exec >> "$LOGFILE" 2>&1

# 使用 tee 命令同时输出到终端和文件
exec > >(tee -a "$LOGFILE") 2>&1

# 推理阶段
echo "[INFO] 开始推理..."
cd ./build

./yolo_obb_track ../config_YOLO11OBB.txt
# ./yolo_obb_track ../config_YOLOV8OBB.txt

echo "[INFO] 推理完成."

# cd ..
# python3 video.py  ./results/imgs_track ./results/output.mp4 --fps 30

# echo "[INFO] 视频生成完成."