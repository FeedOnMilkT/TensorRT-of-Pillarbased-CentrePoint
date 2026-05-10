# TensorRT Deployment of Pillar-Based CenterPoint

> 中文版见 [docs/README.zh-CN.md](docs/README.zh-CN.md)

## Overview

This project is my **undergraduate final-year project and its subsequent improvements**. The goal is to explore an efficient way to deploy a LiDAR-based 3D perception model for real-time inference, using NVIDIA TensorRT as the runtime and CUDA for the data-path stages around it.

The target network is **Pillar-Based CenterPoint** (PointPillars encoder + CenterPoint head, trained under OpenPCDet on nuScenes). Starting from a plain Python reference, the project was progressively rewritten in C++/CUDA and consolidated into a single end-to-end TensorRT engine. Each step in the development log corresponds to one concrete bottleneck that was identified, measured, and removed.

## What it does

- Loads a nuScenes-format point cloud sweep, runs the full detection pipeline, and produces 3D bounding boxes with class scores.
- Runs entirely on the GPU after the first H2D copy: pillarization, PFN, scatter, BEV backbone, detection head, decode + NMS.
- Supports INT8 PTQ calibration against a real nuScenes mini-sweep subset.
- Ships with benchmark scripts that record per-stage timings so each optimization step is measurable.

## Development stages

The history is preserved in the commit log; the table below summarizes the bottleneck removed at each step.

| Stage | What changed | Notes |
|---|---|---|
| 1. CPU preprocess + CPU postprocess, 3-stage engines | Three separate engines (PFN / Scatter / Backbone+Head), pillarization and decode/NMS in C++ on the host | CPU pillarization dominates the per-frame budget |
| 2. CUDA pillarization | Replace the host pillarizer with a CUDA kernel; everything else unchanged | Preprocess no longer the bottleneck; decode/NMS becomes dominant |
| 3. CUDA postprocess (decode + NMS) | CUDA kernels for sigmoid / top-k / box decode / IoU NMS on the head outputs | Still a 3-stage TRT topology with an explicit host-side scatter step |
| 4. Scatter TRT plugin → end-to-end engine | Custom `ScatterPlugin` (CUDA) folded into ONNX so PFN + Scatter + Backbone + Head form a single engine | Removes two H2D/D2H hops and two `enqueueV3` launches per frame |
| 5. INT8 PTQ + pinned memory + double-buffered context | Calibrator built from a curated nuScenes mini-sweep set; pinned host buffers; double execution context to overlap H2D with compute | Final configuration |

> Final mAP / NDS and FPS will be re-benchmarked on a Brev A100 instance and reported here.

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
