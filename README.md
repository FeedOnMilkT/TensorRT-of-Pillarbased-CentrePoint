# TensorRT Deployment of Pillar-Based CenterPoint

> 中文版见 [docs/README.zh-CN.md](docs/README.zh-CN.md)

## Overview

This project is my **undergraduate final-year project and its subsequent improvements**. The goal is to explore an efficient way to deploy a LiDAR-based 3D perception model for real-time inference, using NVIDIA TensorRT as the runtime and CUDA for the data-path stages around it.

The target network is **Pillar-Based CenterPoint** (PointPillars encoder + CenterPoint head, trained under OpenPCDet on nuScenes). The trained checkpoint and ONNX exports consumed here come from my OpenPCDet fork at [FeedOnMilkT/CenterFormer-OpenPCDet-Version](https://github.com/FeedOnMilkT/CenterFormer-OpenPCDet-Version) — use that repo if you want to retrain or re-export the model end-to-end. Starting from a plain Python reference, the project was progressively rewritten in C++/CUDA and consolidated into a single end-to-end TensorRT engine. Each step in the development log corresponds to one concrete bottleneck that was identified, measured, and removed.

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

## Benchmark results

Measured on a single NVIDIA A100-SXM4-80GB (SM 80, driver 580.105, CUDA 12.8, TensorRT 10.8.0.43) inside the `nvcr.io/nvidia/pytorch:25.02-py3` container. Accuracy is the full nuScenes `v1.0-trainval` `val` split (6019 frames) evaluated with the official `nuscenes-devkit` `detection_cvpr_2019` protocol. Latency is end-to-end per frame (H2D copy → pillarize → infer → decode + NMS), median over 1200 iterations with 20-iter warmup, pinned host points, `--cuda-pillarize --cuda-postprocess`.

`4a` = three-engine pipeline (`pfn` → host-orchestrated CUDA scatter → `backbone+head`). `4b` = single end-to-end engine with `PillarScatter` folded in as an `IPluginV3` node inside the ONNX graph.

| Config                    | mAP        | NDS    | mATE   | mAOE   | p50 (ms) | p99 (ms) | FPS    |
| ------------------------- | ---------- | ------ | ------ | ------ | -------- | -------- | ------ |
| FP32 baseline (OpenPCDet) | 0.5003     | 0.6070 | 0.3113 | 0.4292 | –        | –        | –      |
| 4a FP16                   | 0.4378     | 0.5717 | 0.3053 | 0.4334 | 3.11     | 3.76     | 321.3  |
| 4a INT8                   | 0.2491     | 0.4422 | 0.3659 | 0.5441 | 2.31     | 2.82     | 432.4  |
| 4b FP16                   | 0.4384     | 0.5723 | 0.3045 | 0.4315 | 3.09     | 3.58     | 323.4  |
| 4b INT8                   | **broken** | –      | –      | –      | 2.23     | 2.76     | 448.4  |

FP32 baseline numbers are taken directly from the OpenPCDet model-zoo entry for `centerpoint_pillar_nuscenes` and serve as the accuracy ceiling for this checkpoint. Our FP16 runs land ~6 pp below that ceiling; the residual gap is most likely from checkpoint / training-config differences between our exported model and the published OpenPCDet entry, plus minor data-pipeline differences in sweep aggregation — not a TensorRT artifact (FP16 quantization itself typically loses <0.5 mAP). INT8 PTQ shows the expected latency gain (~30 % faster than FP16) but **substantial accuracy loss**: `4a INT8` drops 0.19 mAP / 0.13 NDS off our FP16 number, which is large by INT8 deployment standards (typical PTQ tax is 1-3 mAP).

### Why 4b INT8 fails

`4b INT8` produces zero detections on all 6019 frames despite the engine building successfully. Root cause, in order of contribution:

1. `PillarScatter` is implemented as an `IPluginV3` node inside the ONNX graph. TensorRT 10 INT8 PTQ does not propagate quantization scales across plugin boundaries the way it does across native ops, so the calibration pass cannot capture the true output range of `spatial_features`.
2. The PFN pillar inputs use a dynamic `P` dimension. INT8 calibration must happen at the OPT shape (`P = 12000`), so most frames are padded with zero pillars whose dummy coordinates all map to `grid[0, 0]`. `PillarScatter` accumulates this padding artifact into a single BEV cell with values that do not occur during real inference.
3. Those two effects together corrupt the activation distribution that downstream backbone INT8 layers see during calibration. The calibrated quant scales end up wildly mismatched against the real-inference distribution; by the time signal reaches the detection heads it has been clipped down to a near-constant logit (sigmoid output max ≈ 0.05), and every candidate falls below the score threshold.

This is consistent with the architectural choice in NVIDIA's [`CUDA-CenterPoint`](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-CenterPoint): they deliberately keep `Voxelization` / `Scatter` / `Decode` as CUDA kernels *outside* the TensorRT engine, so the engine itself contains only standard convolutional sub-graphs (`RPN + CenterHead`) that quantize cleanly. The `4a` topology in this project mirrors that choice and is the one that works under PTQ; the `4b` topology was an exploration to see whether folding scatter into the engine was viable in INT8, and the empirical answer here is **no, not under PTQ**.

### Recommendation: prefer QAT over PTQ for INT8

The 0.19 mAP loss on `4a INT8` is too large to ship a downstream perception model on. NVIDIA's reference workflow (`CUDA-CenterPoint/qat`) uses [pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to insert fake-quant nodes in PyTorch, calibrate with a histogram observer, and (for QAT) fine-tune; the resulting ONNX carries explicit Q/DQ nodes, so TensorRT does not have to do its own calibration pass and the dynamic-shape padding pitfall above is bypassed entirely. Their reported PTQ tax is ~0.5 mAP and QAT tax is ~0.35 mAP, both an order of magnitude better than TensorRT-side PTQ as used here.

If a future iteration of this project wants production-grade INT8, the path is:
- adopt the `pytorch-quantization` flow for both `pfn` and `backbone_head`, export ONNX with Q/DQ already embedded; or
- run NVIDIA's sensitivity-profile tool first to identify which layers are quant-sensitive and keep them in FP16.

The `4b` plugin-in-engine topology should be considered a research curiosity until the scatter plugin gains explicit INT8 support (declaring output dynamic range via the IPluginV3 build interface), at which point it can be re-evaluated.

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

### Pre-exported ONNX

`build_engine` reads three ONNX files from `onnx/`. You can either re-export them from PyTorch via the scripts under [scripts/](scripts/) (which need the OpenPCDet fork above), or download the pre-exported copies from Google Drive and drop them into `onnx/`:

- [`pfn.onnx`](https://drive.google.com/file/d/1EH88lgXsDRnDgBb8X0mMzDG6y5eWFqUo/view?usp=drive_link)
- [`backbone_head.onnx`](https://drive.google.com/file/d/1CxWwLJzggHHShk6083CKYzhh2pPSPfzV/view?usp=drive_link)
- [`centerpoint_e2e.onnx`](https://drive.google.com/file/d/1ufGUH7-KOySNc3pBIIYVZQoI0x0EFlcO/view?usp=drive_link)

## Acknowledgements

- **[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)** — the training framework that produced the underlying CenterPoint-PointPillars checkpoint, the dataloader / sweep-aggregation conventions reused by the eval pipeline here, and the published FP32 model-zoo numbers used as the accuracy ceiling. My fork [FeedOnMilkT/CenterFormer-OpenPCDet-Version](https://github.com/FeedOnMilkT/CenterFormer-OpenPCDet-Version) is what was actually used to train and export the model.
- **[NVIDIA-AI-IOT/Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution)** — reference TensorRT deployment for LiDAR detectors (`CUDA-CenterPoint`, `CUDA-BEVFusion`, `CUDA-PointPillars`). Their architectural choice to keep voxelization / scatter / decode as CUDA kernels outside the TensorRT engine, and their `pytorch-quantization`-based QAT workflow under `CUDA-CenterPoint/qat`, directly informed the analysis of the `4b INT8` failure and the QAT recommendation above.
- **[nuScenes](https://www.nuscenes.org/)** — dataset and the official `nuscenes-devkit` evaluation protocol used to report mAP / NDS.
