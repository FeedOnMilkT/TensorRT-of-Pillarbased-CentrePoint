# Pillar-Based CenterPoint 的 TensorRT 部署

> English: [../README.md](../README.md)

## 项目简介

这是我的**本科毕业设计以及后续的改进工作**，目标是探索一种高效部署激光雷达三维感知模型的方法：使用 NVIDIA TensorRT 作为推理后端，并用 CUDA 重写数据通路两端的预处理与后处理。

目标网络是 **Pillar-Based CenterPoint**（PointPillars 编码器 + CenterPoint 检测头，在 OpenPCDet 框架下使用 nuScenes 数据集训练）。项目从一个纯 Python 的参考实现出发，逐步用 C++/CUDA 改写，并最终合并为单一的端到端 TensorRT 引擎。开发日志里每一步都对应一个具体被定位、测量、并消除的瓶颈。

## 功能

- 读取一帧 nuScenes 格式的点云 sweep，跑完整检测流程，输出 3D bbox 与类别得分。
- 第一次 H2D 拷贝之后全程在 GPU 上：pillarize、PFN、scatter、BEV backbone、检测头、decode + NMS。
- 支持基于 nuScenes mini-sweep 子集的 INT8 PTQ 校准。
- 提供带分阶段计时的 benchmark 脚本，使每一次优化都可量化。

## 开发阶段

提交历史已经记录了完整流程；下表汇总每一步消除的瓶颈。

| 阶段 | 改动 | 说明 |
|---|---|---|
| 1. CPU 预处理 + CPU 后处理，三段式引擎 | 三个独立引擎（PFN / Scatter / Backbone+Head），pillarize 和 decode/NMS 都在 host C++ 中跑 | CPU pillarize 占据了每帧的主要开销 |
| 2. CUDA pillarize | 把 host 端的 pillarize 替换成 CUDA 实现，其它保持不变 | 预处理不再是瓶颈，decode/NMS 成为主要开销 |
| 3. CUDA 后处理（decode + NMS） | head 输出的 sigmoid / top-k / box decode / IoU NMS 全部 CUDA kernel 化 | 仍为三段式 TRT 拓扑，需要显式 host-side scatter |
| 4. Scatter TRT plugin → 端到端引擎 | 把 CUDA scatter 封装成 `ScatterPlugin` 并塞回 ONNX，PFN + Scatter + Backbone + Head 合成单一引擎 | 每帧少两次 H2D/D2H 与两次 `enqueueV3` |
| 5. INT8 PTQ + pinned memory + 双 context | 用筛选过的 nuScenes mini-sweep 做 INT8 校准；host 端 pinned buffer；双执行 context 让 H2D 与计算 overlap | 最终配置 |

> 最终的 mAP / NDS 与 FPS 会在 Brev A100 实例上重新 benchmark 后补充。

## 目录结构

```
include/        头文件（engine builder、infer context、plugin、kernels）
src/            host 代码（pipeline 胶水层、CPU 参考实现、calibrator、单测）
cuda/           CUDA kernel（pillarize、scatter、postprocess）
plugin/         TensorRT ScatterPlugin（IPluginV2DynamicExt）
pipeline/       端到端推理可执行（infer_pipeline.cpp）
onnx/           导出的 ONNX（pfn / backbone_head / centerpoint_e2e）
engines/        构建出来的 TRT 引擎（实际产物已 gitignore）
scripts/        ONNX 导出 / 拼接 / 评测 / benchmark
results/        各阶段 benchmark JSON 与 nuScenes 评测产物
```

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

依赖：TensorRT（≥ 8.6）、CUDA、与 TensorRT 版本匹配的 `nvcc`。ONNX 导出和评测脚本还需要一个含 `torch`、`onnx`、`nuscenes-devkit` 的 Python 环境。
