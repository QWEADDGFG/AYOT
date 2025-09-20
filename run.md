测试用：
/home/HwHiAiUser/gp/AYOT/testdata

<!-- ============================== -->
python resize_and_back.py

==================================================
图像处理工具
==================================================
1. 处理图像（缩放并填充到目标尺寸）
2. 恢复图像（恢复到原始尺寸）
3. 查看元数据信息
4. 退出
--------------------------------------------------
请选择功能 (1-4): 1
开始处理图像...
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000008.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000004.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000010.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000007.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000000.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000009.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000002.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000001.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000006.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000003.jpg
Processed and saved /home/HwHiAiUser/gp/AYOT/testdata/640/000005.jpg
Metadata saved to ./resize_metadata.json
图像处理完成！

==================================================
图像处理工具
==================================================
1. 处理图像（缩放并填充到目标尺寸）
2. 恢复图像（恢复到原始尺寸）
3. 查看元数据信息
4. 退出
--------------------------------------------------
请选择功能 (1-4): 3

找到 11 个图像的元数据：
  000008.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000004.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000010.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000007.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000000.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000009.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000002.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000001.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000006.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000003.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]
  000005.jpg:
    原始尺寸: [800, 500]
    缩放尺寸: [640, 400]
    缩放比例: 0.8000
    目标尺寸: [640, 640]

==================================================
图像处理工具
==================================================
1. 处理图像（缩放并填充到目标尺寸）
2. 恢复图像（恢复到原始尺寸）
3. 查看元数据信息
4. 退出
--------------------------------------------------
请选择功能 (1-4): 4
退出程序。


/home/HwHiAiUser/gp/AYOT/testdata/640

./run_build.sh
[INFO] 开始编译源代码...
-- set INC_PATH: /usr/local/Ascend/ascend-toolkit/latest
-- set LIB_PATH: /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
-- set THIRDPART: /usr/local/Ascend/ascend-toolkit/latest/thirdpart
-- Configuring done
-- Generating done
-- Build files have been written to: /home/HwHiAiUser/gp/AYOT/build
[ 12%] Building CXX object CMakeFiles/yolo_obb_track.dir/main_all.cpp.o
[ 25%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/BYTETracker_obb.cpp.o
[ 37%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/kalmanFilter_obb.cpp.o
[ 50%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/STrack_obb.cpp.o
[ 62%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/lapjv.cpp.o
[ 75%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/utils_obb.cpp.o
[ 87%] Building CXX object CMakeFiles/yolo_obb_track.dir/src/yolo_obb.cpp.o
[100%] Linking CXX executable yolo_obb_track
[100%] Built target yolo_obb_track
[INFO] 编译完成.


cd /home/HwHiAiUser/gp/AYOT/build

./yolo_obb_track ../config_YOLO11OBB.txt
Loaded config:
  Task: track
  Model: /home/HwHiAiUser/gp/AYOT/model/YOLO11s_guyu_obb_MVRSD_640.om
  Model Type: YOLO11_OBB
  Input: /home/HwHiAiUser/gp/AYOT/testdata/640
  Class num: 5
  Labels: AFV, CV, LMV, MCV, SMV

[INFO]  Acl init ok
[INFO]  Open device 0 ok
[INFO]  Use default context currently
[INFO]  dvpp init resource ok
[INFO]  Load model /home/HwHiAiUser/gp/AYOT/model/YOLO11s_guyu_obb_MVRSD_640.om success
[INFO]  Create model description success
[INFO]  Create model(/home/HwHiAiUser/gp/AYOT/model/YOLO11s_guyu_obb_MVRSD_640.om) output success
[INFO]  Init model /home/HwHiAiUser/gp/AYOT/model/YOLO11s_guyu_obb_MVRSD_640.om success
[INFO]  Model initialized successfully
[INFO]  Input size: 640x640
[INFO]  Output boxes: 8400
[INFO]  Classes: 5
[INFO]  Using ProbIoU for NMS processing
Init ByteTrack!
[Track] 检测模块平均 FPS = 26.6406  |  跟踪模块平均 FPS = 79.4388
[INFO]  Unload model /home/HwHiAiUser/gp/AYOT/model/YOLO11s_guyu_obb_MVRSD_640.om success
[INFO]  destroy context ok
[INFO]  Reset device 0 ok
[INFO]  Finalize acl ok


python resize_and_back.py

==================================================
图像处理工具
==================================================
1. 处理图像（缩放并填充到目标尺寸）
2. 恢复图像（恢复到原始尺寸）
3. 查看元数据信息
4. 退出
--------------------------------------------------
请选择功能 (1-4): 2
开始恢复图像...
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000008.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000004.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000010.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000007.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000009.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000002.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000001.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000006.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000003.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
Restored and saved /home/HwHiAiUser/gp/AYOT/results/imgs_no_bg/000005.jpg
  Original size: [800, 500] -> Restored size: (800, 500)
图像恢复完成！

==================================================
图像处理工具
==================================================
1. 处理图像（缩放并填充到目标尺寸）
2. 恢复图像（恢复到原始尺寸）
3. 查看元数据信息
4. 退出
--------------------------------------------------
请选择功能 (1-4): 4
退出程序。
