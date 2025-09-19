#!/usr/bin/env python3
import os
from PIL import Image

SRC_DIR = '/home/HwHiAiUser/gp/DATASETS/testv2'
DST_DIR = '/home/HwHiAiUser/gp/DATASETS/testv2_320_02'
TARGET_SIZE = (320, 320)

def resize_keep_aspect(img: Image.Image) -> Image.Image:
    """等比例缩放+左上角对齐填充到目标尺寸"""
    w, h = img.size
    
    # 根据参考代码的逻辑计算缩放比例
    if w >= h:
        # 宽度大于等于高度，以宽度为准缩放
        resize_scale = w / TARGET_SIZE[0]
        new_w = TARGET_SIZE[0]
        new_h = int(h / resize_scale)
    else:
        # 高度大于宽度，以高度为准缩放
        resize_scale = h / TARGET_SIZE[1]
        new_h = TARGET_SIZE[1]
        new_w = int(w / resize_scale)
    
    # 缩放图像
    resized_img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 创建目标尺寸的黑色背景图像
    new_img = Image.new('RGB', TARGET_SIZE, (0, 0, 0))  # 黑色背景，对应cv::Mat::zeros
    
    # 将缩放后的图像粘贴到左上角 (0, 0)
    new_img.paste(resized_img, (0, 0))
    
    return new_img

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    for name in os.listdir(SRC_DIR):
        if name.lower().endswith(supported):
            src_path = os.path.join(SRC_DIR, name)
            dst_path = os.path.join(DST_DIR, name)
            with Image.open(src_path) as im:
                im = im.convert('RGB')          # 统一通道数
                new_im = resize_keep_aspect(im)
                new_im.save(dst_path, quality=95)
            print(f'Saved {dst_path}')

if __name__ == '__main__':
    main()