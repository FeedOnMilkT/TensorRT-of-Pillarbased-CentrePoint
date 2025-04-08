cmake_minimum_required(VERSION 3.18)
project(centerpoint_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)  # A100=80, V100=70, H100=90

# TRT 头文件和库路径（容器内实际路径，标准系统位置）
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
)
target_link_libraries(build_engine
    nvinfer nvonnxparser nvinfer_plugin cudart
)

# 推理 pipeline（Phase 4 实现后启用）
# add_executable(infer_pipeline
#     pipeline/infer_pipeline.cpp
#     src/infer_context.cpp
#     cuda/scatter_kernel.cu
# )
# target_link_libraries(infer_pipeline
#     nvinfer nvinfer_plugin cudart
# )
