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

# Plugin 静态库：由 build_engine / infer_pipeline_4b / 单测共享链接
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

add_executable(infer_pipeline
    pipeline/infer_pipeline.cpp
    src/infer_context.cpp
    src/pillarize.cpp
    src/postprocess.cpp
    src/bench.cpp
    cuda/scatter_kernel.cu
    cuda/pillarize_kernel.cu
)
target_link_libraries(infer_pipeline
    nvinfer nvinfer_plugin cudart
)

# 4b 端到端 pipeline：单引擎含 PillarScatter plugin
add_executable(infer_pipeline_4b
    pipeline/infer_pipeline_4b.cpp
    src/infer_context.cpp
    src/pillarize.cpp
    src/postprocess.cpp
    src/bench.cpp
    cuda/pillarize_kernel.cu
)
target_link_libraries(infer_pipeline_4b
    centerpoint_plugin nvinfer nvinfer_plugin cudart
)
