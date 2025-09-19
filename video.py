import cv2
import os
import argparse
import glob
import re

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将图片序列转换为视频')
    parser.add_argument('input_dir', help='输入图片目录路径')
    parser.add_argument('output_video', help='输出视频文件路径')
    parser.add_argument('--fps', type=int, default=25, help='视频帧率 (默认: 25)')
    parser.add_argument('--codec', default='mp4v', help='视频编码器 (默认: mp4v)')
    
    args = parser.parse_args()
    
    IMG_DIR = args.input_dir
    OUT_VIDEO = args.output_video
    
    # 检查输入目录是否存在
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"输入目录不存在: {IMG_DIR}")
    
    # 自动检测图片文件
    # 支持常见的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    img_paths = []
    
    for ext in image_extensions:
        img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))
        img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext.upper())))
    
    if not img_paths:
        raise FileNotFoundError(f"在目录 {IMG_DIR} 中未找到图片文件")
    
    # 按文件名中的数字排序
    def extract_number(filename):
        # 提取文件名中的数字进行排序
        basename = os.path.basename(filename)
        numbers = re.findall(r'\d+', basename)
        return int(numbers[0]) if numbers else 0
    
    img_paths.sort(key=extract_number)
    
    print(f"找到 {len(img_paths)} 张图片")
    print(f"第一张: {os.path.basename(img_paths[0])}")
    print(f"最后一张: {os.path.basename(img_paths[-1])}")
    
    # 读取所有图片
    frames = []
    for i, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"警告: 无法读取图片 {path}")
            continue
        frames.append(img)
        
        # 显示进度
        if (i + 1) % 50 == 0 or i == len(img_paths) - 1:
            print(f"已读取 {i + 1}/{len(img_paths)} 张图片")
    
    if not frames:
        raise ValueError("没有成功读取任何图片")
    
    # 获取尺寸
    h, w = frames[0].shape[:2]
    print(f"视频尺寸: {w}x{h}")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(OUT_VIDEO)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(OUT_VIDEO, fourcc, fps=args.fps, frameSize=(w, h))
    
    if not vw.isOpened():
        raise RuntimeError(f"无法创建视频文件: {OUT_VIDEO}")
    
    # 写入帧
    for i, frame in enumerate(frames):
        # 确保所有帧尺寸一致
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        
        vw.write(frame)
        
        # 显示进度
        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"已写入 {i + 1}/{len(frames)} 帧")
    
    vw.release()
    print(f"视频已生成: {OUT_VIDEO}")
    print(f"总帧数: {len(frames)}, 帧率: {args.fps} fps, 时长: {len(frames)/args.fps:.2f} 秒")

if __name__ == "__main__":
    main()