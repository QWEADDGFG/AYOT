## AYOT

### 0. 介绍

实现OBB数据集的检测及跟踪，基于Ascend310平台，使用YOLOOBB检测模型及ByteTrackOBB跟踪模型。

### 1. 准备环境、数据集、模型

#### 1.1 环境准备
```bash
npu-smi info
+--------------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                                 Version: 23.0.rc3                                     |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B1                 | OK              | 9.0          57                15    / 15            |
| 0       0                     | NA              | 0            7356 / 11577                            |
+===============================+=================+======================================================+

```
#### 1.2 数据集准备
主要是数据集输入与模型期望输入的不一致，建议还是离线处理，其避免了创造新的过程文件，减少了对实时性的损耗。

将数据集尺寸转换为模型期望的代码: resize_and_back.py
使用流程：
1. 配置SRC_DIR与DST_DIR，SRC_DIR为原始数据集路径，DST_DIR为处理后数据集路径。
2. 配置TARGET_SIZE为模型期望输入尺寸。
3. 配置METADATA_FILE为resize_metadata.json文件路径。
4. 运行resize_and_back.py。（选择1）
5. 在模型推理完成后，如果需要将推理后的图像缩放为原尺寸，
则需要配置DST_DIR为推理后的图像，RESTORE_DIR为推理后的图像还原为原始尺寸的图像路径。
6. 运行resize_and_back.py。（选择2）
```python
SRC_DIR = '/home/HwHiAiUser/gp/DATASETS/test0920'
DST_DIR = '/home/HwHiAiUser/gp/AYOT/results/imgs'
RESTORE_DIR = '/home/HwHiAiUser/gp/AYOT/results/imgs_no_bg'
TARGET_SIZE = (640, 640)
METADATA_FILE = os.path.join('./', 'resize_metadata.json')
```

如果不想进行离线处理，代码会在main_all.cpp中的applyLetterbox进行处理，但是会导致推理速度变慢。

#### 1.3 模型准备
**pt-->onnx**
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('/data/gy/gp/Huawei/yolo11obb/runs/train/yolo11sobb_MVRSD2/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=11, dynamic=True, imgsz=640, nms=False)
```
**onnx-->om**
```bash
atc --model=YOLO11s_base_obb_MVRSD_640.onnx --framework=5 --output=YOLO11s_base_obb_MVRSD_640 --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp640.cfg
atc --model=YOLO11s_guyu_obb_MVRSD_640.onnx --framework=5 --output=YOLO11s_guyu_obb_MVRSD_640 --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp640.cfg
```
#### 1.4 Config配置文件
**config_YOLO11OBB.txt**
```txt
# 任务配置
# task=detect
task=track
model_path=/home/HwHiAiUser/gp/AYOT/model/YOLO11s_guyu_obb_MVRSD_640.om
# image_input=/home/HwHiAiUser/gp/DATASETS/testv2_320_02
image_input=/home/HwHiAiUser/gp/DATASETS/testv2
image_output=../results/imgs
label_output=../results/txts 
track_image_output=../results/imgs_track 

# 模型参数
class_num=5
conf_thresh=0.25
nms_thresh=0.45
model_width=640
model_height=640
model_box_num=8400

# 模型类型配置 可选值: YOLO11_OBB, YOLOV8_OBB
model_type=YOLO11_OBB

# 类别标签文件路径
classes_file=/home/HwHiAiUser/gp/AYOT/classes.txt

```
### 2. 编译
```bash
chmod +x run_build.sh
./run_build.sh
```

### 3. 运行
```bash
chmod +x infer_obb_track_config.sh
./infer_obb_track_config.sh
```

### 4. 样例目录结构：
```bash
tree
.
├── CMakeLists.txt
├── LICENSE
├── Readme.md
├── build
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   │   ├── 3.22.1
│   │   │   ├── CMakeCCompiler.cmake
│   │   │   ├── CMakeCXXCompiler.cmake
│   │   │   ├── CMakeDetermineCompilerABI_C.bin
│   │   │   ├── CMakeDetermineCompilerABI_CXX.bin
│   │   │   ├── CMakeSystem.cmake
│   │   │   ├── CompilerIdC
│   │   │   │   ├── CMakeCCompilerId.c
│   │   │   │   ├── a.out
│   │   │   │   └── tmp
│   │   │   └── CompilerIdCXX
│   │   │       ├── CMakeCXXCompilerId.cpp
│   │   │       ├── a.out
│   │   │       └── tmp
│   │   ├── CMakeDirectoryInformation.cmake
│   │   ├── CMakeOutput.log
│   │   ├── CMakeTmp
│   │   ├── Makefile.cmake
│   │   ├── Makefile2
│   │   ├── TargetDirectories.txt
│   │   ├── cmake.check_cache
│   │   ├── progress.marks
│   │   └── yolo_obb_track.dir
│   │       ├── DependInfo.cmake
│   │       ├── build.make
│   │       ├── cmake_clean.cmake
│   │       ├── compiler_depend.internal
│   │       ├── compiler_depend.make
│   │       ├── compiler_depend.ts
│   │       ├── depend.make
│   │       ├── flags.make
│   │       ├── link.txt
│   │       ├── main_all.cpp.o
│   │       ├── main_all.cpp.o.d
│   │       ├── progress.make
│   │       └── src
│   │           ├── BYTETracker_obb.cpp.o
│   │           ├── BYTETracker_obb.cpp.o.d
│   │           ├── STrack_obb.cpp.o
│   │           ├── STrack_obb.cpp.o.d
│   │           ├── kalmanFilter_obb.cpp.o
│   │           ├── kalmanFilter_obb.cpp.o.d
│   │           ├── lapjv.cpp.o
│   │           ├── lapjv.cpp.o.d
│   │           ├── utils_obb.cpp.o
│   │           ├── utils_obb.cpp.o.d
│   │           ├── yolo_obb.cpp.o
│   │           └── yolo_obb.cpp.o.d
│   ├── Makefile
│   ├── cmake_install.cmake
│   └── yolo_obb_track
├── classes.txt
├── config_YOLO11OBB.txt
├── config_YOLOV8OBB.txt
├── git.sh
├── include
│   ├── BYTETracker_obb.h
│   ├── STrack_obb.h
│   ├── dataType_obb.h
│   ├── kalmanFilter_obb.h
│   ├── lapjv.h
│   └── yolo_obb.h
├── infer_obb_track.sh
├── main_all.cpp
├── model
│   ├── YOLO11s_base_obb_MVRSD_640.om
│   ├── YOLO11s_guyu_obb_MVRSD_640.om
│   └── aipp640.cfg
├── resize_and_back.py
├── resize_metadata.json
├── results
│   ├── imgs
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── 000004.jpg
│   │   ├── 000005.jpg
│   │   ├── 000006.jpg
│   │   ├── 000007.jpg
│   │   ├── 000008.jpg
│   │   ├── 000009.jpg
│   │   └── 000010.jpg
│   ├── imgs_no_bg
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── 000004.jpg
│   │   ├── 000005.jpg
│   │   ├── 000006.jpg
│   │   ├── 000007.jpg
│   │   ├── 000008.jpg
│   │   ├── 000009.jpg
│   │   └── 000010.jpg
│   ├── imgs_track
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── 000004.jpg
│   │   ├── 000005.jpg
│   │   ├── 000006.jpg
│   │   ├── 000007.jpg
│   │   ├── 000008.jpg
│   │   ├── 000009.jpg
│   │   └── 000010.jpg
│   └── txts
│       ├── 000001.txt
│       ├── 000002.txt
│       ├── 000003.txt
│       ├── 000004.txt
│       ├── 000005.txt
│       ├── 000006.txt
│       ├── 000007.txt
│       ├── 000008.txt
│       ├── 000009.txt
│       └── 000010.txt
├── run.md
├── run_build.sh
├── src
│   ├── BYTETracker_obb.cpp
│   ├── STrack_obb.cpp
│   ├── kalmanFilter_obb.cpp
│   ├── lapjv.cpp
│   ├── utils_obb.cpp
│   └── yolo_obb.cpp
├── testdata
│   ├── 640
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── 000004.jpg
│   │   ├── 000005.jpg
│   │   ├── 000006.jpg
│   │   ├── 000007.jpg
│   │   ├── 000008.jpg
│   │   ├── 000009.jpg
│   │   └── 000010.jpg
│   └── back
│       ├── 000000.jpg
│       ├── 000001.jpg
│       ├── 000002.jpg
│       ├── 000003.jpg
│       ├── 000004.jpg
│       ├── 000005.jpg
│       ├── 000006.jpg
│       ├── 000007.jpg
│       ├── 000008.jpg
│       ├── 000009.jpg
│       └── 000010.jpg
└── video.py

21 directories, 135 files
```

