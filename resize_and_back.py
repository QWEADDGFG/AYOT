#!/usr/bin/env python3
import os
import json
from PIL import Image

SRC_DIR = '/home/HwHiAiUser/gp/AYOT/testdata/back'
# DST_DIR = '/home/HwHiAiUser/gp/AYOT/testdata/640'
DST_DIR = '/home/HwHiAiUser/gp/AYOT/results/imgs_track'
RESTORE_DIR = '/home/HwHiAiUser/gp/AYOT/results/imgs_no_bg'
TARGET_SIZE = (640, 640)
METADATA_FILE = os.path.join('./', 'resize_metadata.json')

def resize_keep_aspect(img: Image.Image, filename: str, metadata: dict) -> Image.Image:
    """等比例缩放+左上角对齐填充到目标尺寸，不裁切，并记录变换信息"""
    w, h = img.size
    original_size = (w, h)
    
    # 计算两个方向的缩放比例，选择较小的那个以避免裁切
    scale_w = TARGET_SIZE[0] / w
    scale_h = TARGET_SIZE[1] / h
    scale = min(scale_w, scale_h)  # 选择较小的缩放比例确保不裁切
    
    # 计算缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized_img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 创建目标尺寸的图像
    new_img = Image.new('RGB', TARGET_SIZE, (114, 114, 114))  # 灰色背景
    
    # 将缩放后的图像粘贴到左上角 (0, 0)
    new_img.paste(resized_img, (0, 0))
    
    # 保存变换信息
    metadata[filename] = {
        'original_size': original_size,
        'scaled_size': (new_w, new_h),
        'scale_factor': scale,
        'target_size': TARGET_SIZE
    }
    
    return new_img

def restore_original_size(processed_img: Image.Image, metadata_info: dict) -> Image.Image:
    """将处理后的图像恢复到原始尺寸"""
    original_size = metadata_info['original_size']
    scaled_size = metadata_info['scaled_size']
    scale_factor = metadata_info['scale_factor']
    
    # 从处理后的图像中提取有效区域（左上角的缩放图像）
    scaled_region = processed_img.crop((0, 0, scaled_size[0], scaled_size[1]))
    
    # 计算逆向缩放比例
    inverse_scale = 1.0 / scale_factor
    
    # 恢复到原始尺寸
    restored_img = scaled_region.resize(original_size, Image.LANCZOS)
    
    return restored_img

def process_images():
    """处理图像并保存元数据"""
    os.makedirs(DST_DIR, exist_ok=True)
    metadata = {}
    
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    for name in os.listdir(SRC_DIR):
        if name.lower().endswith(supported):
            src_path = os.path.join(SRC_DIR, name)
            dst_path = os.path.join(DST_DIR, name)
            
            with Image.open(src_path) as im:
                im = im.convert('RGB')          # 统一通道数
                new_im = resize_keep_aspect(im, name, metadata)
                new_im.save(dst_path, quality=95)
            print(f'Processed and saved {dst_path}')
    
    # 保存元数据
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f'Metadata saved to {METADATA_FILE}')

def restore_images():
    """恢复图像到原始尺寸"""
    os.makedirs(RESTORE_DIR, exist_ok=True)
    
    # 读取元数据
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file {METADATA_FILE} not found. Please run process_images() first.")
        return
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    for name in os.listdir(DST_DIR):
        if name.lower().endswith(supported) and name in metadata:
            processed_path = os.path.join(DST_DIR, name)
            restore_path = os.path.join(RESTORE_DIR, f'{name}')
            
            with Image.open(processed_path) as processed_img:
                restored_img = restore_original_size(processed_img, metadata[name])
                restored_img.save(restore_path, quality=95)
            print(f'Restored and saved {restore_path}')
            print(f'  Original size: {metadata[name]["original_size"]} -> Restored size: {restored_img.size}')

def main():
    """主函数，提供选择功能"""
    while True:
        print("\n" + "="*50)
        print("图像处理工具")
        print("="*50)
        print("1. 处理图像（缩放并填充到目标尺寸）")
        print("2. 恢复图像（恢复到原始尺寸）")
        print("3. 查看元数据信息")
        print("4. 退出")
        print("-"*50)
        
        choice = input("请选择功能 (1-4): ").strip()
        
        if choice == '1':
            print("开始处理图像...")
            process_images()
            print("图像处理完成！")
            
        elif choice == '2':
            print("开始恢复图像...")
            restore_images()
            print("图像恢复完成！")
            
        elif choice == '3':
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"\n找到 {len(metadata)} 个图像的元数据：")
                for filename, info in metadata.items():
                    print(f"  {filename}:")
                    print(f"    原始尺寸: {info['original_size']}")
                    print(f"    缩放尺寸: {info['scaled_size']}")
                    print(f"    缩放比例: {info['scale_factor']:.4f}")
                    print(f"    目标尺寸: {info['target_size']}")
            else:
                print("未找到元数据文件，请先处理图像。")
                
        elif choice == '4':
            print("退出程序。")
            break
            
        else:
            print("无效选择，请输入 1-4。")

if __name__ == '__main__':
    main()