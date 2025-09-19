## 目录结构
```bash
tree
.
├── CMakeLists.txt
├── LICENSE
├── Readme.md
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
│   ├── aipp640.cfg
│   └── best0919.om
├── resize.py
├── run_build.sh
├── src
│   ├── BYTETracker_obb.cpp
│   ├── STrack_obb.cpp
│   ├── kalmanFilter_obb.cpp
│   ├── lapjv.cpp
│   ├── utils_obb.cpp
│   └── yolo_obb.cpp
└── video.py

3 directories, 23 files

```

## 编译说明

### 1. 准备环境
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

### 2. 编译
```bash
chmod +x run_build.sh
./run_build.sh
```
```bash
./run_build.sh
[INFO] 开始编译源代码...
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenCV: /usr/local (found version "4.5.4") 
-- set INC_PATH: /usr/local/Ascend/ascend-toolkit/latest
-- set LIB_PATH: /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
-- set THIRDPART: /usr/local/Ascend/ascend-toolkit/latest/thirdpart
-- Configuring done
-- Generating done
-- Build files have been written to: /home/HwHiAiUser/gp/AYOT/build
[ 12%] Building CXX object CMakeFiles/yolo_obb_track.dir/main_all.cpp.o
[ 25%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/STrack_obb.cpp.o
[ 37%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/BYTETracker_obb.cpp.o
[ 50%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/kalmanFilter_obb.cpp.o
[ 62%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/lapjv.cpp.o
[ 75%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/utils_obb.cpp.o
[ 87%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/yolo_obb.cpp.o
[100%] Linking CXX executable yolo_obb_track
[100%] Built target yolo_obb_track
[INFO] 编译完成.

```
### 3. 运行
```bash
chmod +x infer_obb_track.sh
./infer_obb_track.sh
```

