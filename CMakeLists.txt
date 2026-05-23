cmake_minimum_required(VERSION 3.18)
project(centerpoint_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# CUDA arch 写死为 80（A100 SM_80）。
# 切换硬件时显式覆盖：cmake -DCMAKE_CUDA_ARCHITECTURES=70（V100）/90（H100）。
# 不做自动检测：项目目前只在 sidney-dev / Brev A100 上跑，写死 80 让生成的
# fatbin 不携带多余 cubin，缩短链接体积，也避免 nvcc 报告含糊的"unsupported gpu arch"。
set(CMAKE_CUDA_ARCHITECTURES 80)  # A100=80, V100=70, H100=90

# TRT 头文件和库路径（sidney-dev 容器内 TRT 10.8 实测路径，标准系统位置）。
# 路径外的安装位置请通过 -DTRT_INCLUDE_DIR=... -DTRT_LIB_DIR=... 覆盖。
set(TRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include dir")
set(TRT_LIB_DIR     "/usr/lib/x86_64-linux-gnu"     CACHE PATH "TensorRT lib dir")
find_package(CUDA REQUIRED)

include_directories(
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(${TRT_LIB_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_executable(build_engine
    src/engine_builder.cpp
    src/main.cpp
    src/calibrator.cpp
)
target_link_libraries(build_engine
    centerpoint_plugin nvinfer nvonnxparser nvinfer_plugin cudart
)

add_executable(test_infer
    src/infer_context.cpp
    src/test_infer.cpp
)
target_link_libraries(test_infer
    nvinfer nvinfer_plugin cudart
)

add_executable(test_scatter
    cuda/scatter_kernel.cu
    src/test_scatter.cpp
)
target_link_libraries(test_scatter cudart)

add_executable(test_pillarize
    cuda/pillarize_kernel.cu
    src/pillarize.cpp
    src/test_pillarize.cpp
)
target_link_libraries(test_pillarize cudart)

add_executable(test_postprocess
    cuda/postprocess_kernel.cu
    src/postprocess.cpp
    src/test_postprocess.cpp
)
target_link_libraries(test_postprocess cudart)

# Plugin 静态库：由 build_engine / infer_pipeline (4b mode) / 单测共享链接
add_library(centerpoint_plugin STATIC
    plugin/scatter_plugin.cpp
    cuda/scatter_kernel.cu
)
target_link_libraries(centerpoint_plugin nvinfer cudart)

add_executable(test_scatter_plugin
    src/test_scatter_plugin.cpp
)
target_link_libraries(test_scatter_plugin
    centerpoint_plugin nvinfer nvinfer_plugin cudart
)

# 统一 pipeline：通过 --mode {4a,4b} 切换
#   4a：pfn + 显式 scatter + backbone+head（scatter_kernel.cu 直接编进 binary）
#   4b：单一 e2e engine（scatter 在图内，由 centerpoint_plugin 提供 IPluginV3）
add_executable(infer_pipeline
    pipeline/infer_pipeline.cpp
    src/infer_context.cpp
    src/pillarize.cpp
    src/postprocess.cpp
    src/bench.cpp
    cuda/pillarize_kernel.cu
    cuda/postprocess_kernel.cu
)
# scatter_kernel.cu 由 centerpoint_plugin 提供，避免多定义
target_link_libraries(infer_pipeline
    centerpoint_plugin nvinfer nvinfer_plugin cudart
)