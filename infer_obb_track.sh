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

# ./yolo_obb_track detect \
#     --model /home/HwHiAiUser/gp/AYOTV2/model/mvrsd_v11sobb_640.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/testv2/000001.jpg \
#     --image_out  ../results/imgs_01 \
#     --label_out  ../results/txts_01

# ./yolo_obb_track detect \
#     --model ../model//home/HwHiAiUser/gp/Ascend_YOLO_OBB_Track/model/mvrsd-obb_v11.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/testv2_800 \
#     --image_out  ../results/imgs \
#     --label_out  ../results/txts


# ./yolo_obb_track detect \
#     --model /home/HwHiAiUser/gp/Ascend_YOLO/model/YOLO11s_base_obb_MVRSD_640.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/testv2 \
#     --image_out  ../results/imgs \
#     --label_out  ../results/txts

# ./yolo_obb_track track \
#     --model /home/HwHiAiUser/gp/AYOTV2/model/best.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/testv2_800 \
#     --image_out  ../results/imgs \
#     --label_out  ../results/txts \
#     --track_image_out ../results/imgs_track 

# ./yolo_obb_track track \
#     --model /home/HwHiAiUser/gp/AYOTV2/model/mvrsd_v11sobb_640.om \
#     --input  /home/HwHiAiUser/gp/DATASETS/testv2 \
#     --image_out  ../results/imgs \
#     --label_out  ../results/txts \
#     --track_image_out ../results/imgs_track 

./yolo_obb_track track \
    --model /home/HwHiAiUser/gp/AYOT/model/best0919.om \
    --input  /home/HwHiAiUser/gp/DATASETS/testv2 \
    --image_out  ../results/imgs \
    --label_out  ../results/txts \
    --track_image_out ../results/imgs_track 


echo "[INFO] 推理完成."

cd ..
python3 video.py  ./results/imgs_track ./results/output.mp4 --fps 30

echo "[INFO] 视频生成完成."