# TensorRT Deployment of Pillar-Based CenterPoint

> **English (default)** · [中文](#中文版)

## Overview

This project is my **undergraduate final-year project and its subsequent improvements**. The goal is to explore an efficient way to deploy a LiDAR-based 3D perception model for real-time inference, using NVIDIA TensorRT as the runtime and CUDA for the data-path stages around it.

The target network is **Pillar-Based CenterPoint** (PointPillars encoder + CenterPoint head, trained under OpenPCDet on nuScenes). Starting from a plain Python reference, the project was progressively rewritten in C++/CUDA and consolidated into a single end-to-end TensorRT engine. Each step in the development log corresponds to one concrete bottleneck that was identified, measured, and removed.

## What it does

- Loads a nuScenes-format point cloud sweep, runs the full detection pipeline, and produces 3D bounding boxes with class scores.
- Runs entirely on the GPU after the first H2D copy: pillarization, PFN, scatter, BEV backbone, detection head, decode + NMS.
- Supports INT8 PTQ calibration against a real nuScenes mini-sweep subset.
- Ships with benchmark scripts that record per-stage timings so each optimization step is measurable.

## Development stages

The history is preserved in the commit log; the table below summarizes the bottleneck removed at each step and the resulting median FPS measured on the same input set / same GPU.

| Stage | What changed | Median FPS | Notes |
|---|---|---|---|
| 1. CPU preprocess + CPU postprocess, 3-stage engines | Three separate engines (PFN / Scatter / Backbone+Head), pillarization and decode/NMS in C++ on the host | **1.02** | CPU pillarization dominates (~900 ms/frame) |
| 2. CUDA pillarization | Replace the host pillarizer with a CUDA kernel; everything else unchanged | **13.71** | Preprocess no longer the bottleneck; decode/NMS now dominant |
| 3. CUDA postprocess (decode + NMS) | CUDA kernels for sigmoid / top-k / box decode / IoU NMS on the head outputs | **47.59** | Still a 3-stage TRT topology with an explicit host-side scatter step |
| 4. Scatter TRT plugin → end-to-end engine | Custom `ScatterPlugin` (CUDA) folded into ONNX so PFN + Scatter + Backbone + Head form a single engine | (E2E topology in place) | Removes two H2D/D2H hops and two `enqueueV3` launches per frame |
| 5. INT8 PTQ + pinned memory + double-buffered context | Calibrator built from a curated nuScenes mini-sweep set; pinned host buffers; double execution context to overlap H2D with compute | **57.69** | Final configuration used for the reported mAP/NDS |

> All numbers are `fps_median` over 200 iterations from `results/bench_*.json` in this repo. The final number (57.69 FPS) is from `results/bench_4b_pinned.json`, the E2E INT8 + pinned + double-buffer configuration.

## Accuracy (nuScenes val)

INT8 engine, evaluated through the official `nuscenes-devkit` after the LiDAR → ego → global frame transform:

| Metric | Value |
|---|---|
| **mAP**  | **0.3896** |
| **NDS**  | **0.4751** |

Source: `results/eval_output/trt_result/metrics_summary.json`.

## Project layout

```
include/        public headers (engine builder, infer context, plugin, kernels)
src/            host code (pipeline glue, pillarize CPU ref, calibrator, tests)
cuda/           CUDA kernels (pillarize, scatter, postprocess)
plugin/         TensorRT ScatterPlugin (IPluginV2DynamicExt)
pipeline/       end-to-end inference binary (infer_pipeline.cpp)
onnx/           exported ONNX (pfn / backbone_head / centerpoint_e2e)
engines/        built TRT engines (gitignored payloads)
scripts/        ONNX export, ONNX stitching, evaluation, benchmarking
results/        per-stage benchmark JSONs and nuScenes evaluation output
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires TensorRT (≥ 8.6), CUDA, and a working `nvcc` matching the TensorRT version. ONNX export and evaluation scripts additionally need a Python environment with `torch`, `onnx`, and `nuscenes-devkit`.

---

## 中文版

## 项目简介

这是我的**本科毕业设计以及后续的改进工作**，目标是探索一种高效部署激光雷达三维感知模型的方法：使用 NVIDIA TensorRT 作为推理后端，并用 CUDA 重写数据通路两端的预处理与后处理。

目标网络是 **Pillar-Based CenterPoint**（PointPillars 编码器 + CenterPoint 检测头，在 OpenPCDet 框架下使用 nuScenes 数据集训练）。项目从一个纯 Python 的参考实现出发，逐步用 C++/CUDA 改写，并最终合并为单一的端到端 TensorRT 引擎。开发日志里每一步都对应一个具体被定位、测量、并消除的瓶颈。

## 功能

- 读取一帧 nuScenes 格式的点云 sweep，跑完整检测流程，输出 3D bbox 与类别得分。
- 第一次 H2D 拷贝之后全程在 GPU 上：pillarize、PFN、scatter、BEV backbone、检测头、decode + NMS。
- 支持基于 nuScenes mini-sweep 子集的 INT8 PTQ 校准。
- 提供带分阶段计时的 benchmark 脚本，使每一次优化都可量化。

## 开发阶段

提交历史已经记录了完整流程；下表汇总每一步消除的瓶颈以及在相同输入 / 相同 GPU 上测得的中位 FPS。

| 阶段 | 改动 | 中位 FPS | 说明 |
|---|---|---|---|
| 1. CPU 预处理 + CPU 后处理，三段式引擎 | 三个独立引擎（PFN / Scatter / Backbone+Head），pillarize 和 decode/NMS 都在 host C++ 中跑 | **1.02** | CPU pillarize 占主要时间（约 900 ms/帧） |
| 2. CUDA pillarize | 把 host 端的 pillarize 替换成 CUDA 实现，其它保持不变 | **13.71** | 预处理不再是瓶颈，decode/NMS 成为主要开销 |
| 3. CUDA 后处理（decode + NMS） | head 输出的 sigmoid / top-k / box decode / IoU NMS 全部 CUDA kernel 化 | **47.59** | 仍为三段式 TRT 拓扑，需要显式 host-side scatter |
| 4. Scatter TRT plugin → 端到端引擎 | 把 CUDA scatter 封装成 `ScatterPlugin` 并塞回 ONNX，PFN + Scatter + Backbone + Head 合成单一引擎 | （已切换到 E2E 拓扑） | 每帧少两次 H2D/D2H 与两次 `enqueueV3` |
| 5. INT8 PTQ + pinned memory + 双 context | 用筛选过的 nuScenes mini-sweep 做 INT8 校准；host 端 pinned buffer；双执行 context 让 H2D 与计算 overlap | **57.69** | 最终配置，对应下文 mAP/NDS |

> 所有数据来自仓库里 `results/bench_*.json` 的 200 次迭代 `fps_median`。最终的 57.69 FPS 来自 `results/bench_4b_pinned.json`，即 E2E INT8 + pinned + 双缓冲配置。

## 精度（nuScenes val）

INT8 引擎，按 LiDAR → ego → global frame 变换后过官方 `nuscenes-devkit` 评测：

| 指标 | 数值 |
|---|---|
| **mAP** | **0.3896** |
| **NDS** | **0.4751** |

来源：`results/eval_output/trt_result/metrics_summary.json`。

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
