import cv2
import os

IMG_DIR  = '/home/HwHiAiUser/gp/AYOTV2/results/imgs_track'
OUT_VIDEO = '/home/HwHiAiUser/gp/AYOTV2/results/imgs_video_trackv2.mp4'

# 1. 按数字顺序读取 000001.jpg … 000413.jpg
img_names = [f'{i:06d}.jpg' for i in range(1, 361)]
frames = []
for name in img_names:
    path = os.path.join(IMG_DIR, name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    frames.append(img)

# 2. 获取尺寸
h, w = frames[0].shape[:2]

# 3. 创建 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 可换 'avc1' 或 'h264'
vw = cv2.VideoWriter(OUT_VIDEO, fourcc, fps=25, frameSize=(w, h))

for f in frames:
    vw.write(f)

vw.release()
print('视频已生成:', OUT_VIDEO)