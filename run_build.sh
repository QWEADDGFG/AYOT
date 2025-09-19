#!/bin/bash
# chmod +x run_build.sh
set -e  # 遇到错误立即退出

BUILD_DIR="./build"
mkdir -p "$BUILD_DIR"

echo "[INFO] 开始编译源代码..."
cd build
cmake ..
make clean
make -j4
echo "[INFO] 编译完成."