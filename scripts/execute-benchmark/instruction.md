# CenterPoint TRT Benchmark — 自动化脚本套件（第二版）

> 第一版（已废弃）假设 Brev 干净实例 + 4 part nuScenes + TRT 8.6 + `/home/uceeanz/...` 路径。
> 第二版基于当前共享容器 `sidney-dev`（TRT 10 / CUDA 12.8 / nvcr pytorch 25.02），数据走全量 nuScenes，路径全部重锚定。

---

## ⚠️ 任何 session 开工前必读（按顺序）

1. **[/workspace/CLAUDE.md](/workspace/CLAUDE.md)** — 容器环境全局规范：哪些路径持久、哪些是 ephemeral；`/workspace` 是 193G 小盘禁止存数据；`/data/sidney/` 才是 19TB 大盘；GPU 共享给团队需礼让；`pip install` 落 ephemeral 层。
2. **[/workspace/demo/TensorRT-of-Pillarbased-CentrePoint/CLAUDE.md](/workspace/demo/TensorRT-of-Pillarbased-CentrePoint/CLAUDE.md)** — 项目架构：4a/4b 两套 pipeline、PillarScatter 插件契约、INT8 PTQ 流程、动态 shape profile、当前硬编码状况。
3. **本文件全文**。

不读这三份就动手 = 99% 概率把数据下到小盘、把 pip 包装丢、或者按已废弃的 TRT 8.6 假设写代码。

---

## Context（环境快照）

- **位置**：容器 `sidney-dev`，工作目录 `/workspace/demo/TensorRT-of-Pillarbased-CentrePoint/`（容器内）。本项目代码全在这里。
- **运行时**：TensorRT 10、CUDA 12.8、PyTorch（NGC 25.02）。`tensorrt`、`torch`、`onnx_graphsurgeon` 已可直接 import。
- **GPU**：2× A100-80GB，**与团队共享**。跑 benchmark 前 `nvidia-smi` 看一眼，避开他人作业。
- **数据盘**：`/data/sidney/datasets/`（19TB RAID，挂载持久）。**所有 nuScenes 原始数据、sweep 缓存、模型权重必须落这里。**
- **项目盘**：`/workspace/`（193G，与团队 home 共享）。**只放代码、ONNX/engine 小文件、benchmark 结果（< 1GB）。**
- **Ephemeral**：`pip install` 默认落到容器层，重启容器会丢；本套件**直接用容器自带 python**，装 OpenPCDet / spconv / nuscenes-devkit 都进容器层（不建 venv）。重跑 benchmark 前若容器被 `docker rm` 重建过，重跑 `02b` 的 `setup_openpcdet_env.sh` 会自动再装一遍，可接受。

---

## 配置基线（用户已决策，勿改）

| 项 | 值 |
|---|---|
| 数据 | **全量 nuScenes v1.0-trainval（10 part）+ mini** |
| 数据根目录 | **`/data/sidney/datasets/nuscenes/`** |
| sweep 缓存目录 | `/data/sidney/datasets/nuscenes_sweeps_cache/` |
| val 子集 | 全 val（~6019 帧，mAP 抖动应 < 0.3%） |
| 算力档 | `a100_full` + `mig_1g10gb`（**不锁频**） |
| 评测路径 | 只测 `--cuda-pillarize --cuda-postprocess`（CPU 路径不测） |
| 精度 | TRT FP16 + TRT INT8（FP32 不导 engine） |
| FP32 baseline | OpenPCDet PyTorch（**仅 02b 用**，作精度上限） |
| 脚本目录 | **`scripts/execute-benchmark/`**（即本文件所在目录） |
| 项目根记号 | `$REPO_ROOT` = `/workspace/demo/TensorRT-of-Pillarbased-CentrePoint` |

---

## Phase 0 — 硬编码审计（**必须串行先做，单 session 完成**）

老代码里散布着大量硬编码路径，按当前环境直接跑会全线崩溃。Phase 1 写脚本之前必须先把项目调通到"参数化可移植"的状态。这步不并行，因为它会跨文件改 C++ / Python / CMake，三个并行 session 同时改必冲突。

### 已知必须审计的点（非穷举，做这步的 session 自己 grep 兜底）

| 文件 | 硬编码内容 | 期望处理 |
|---|---|---|
| [src/main.cpp](../../src/main.cpp) | `/workspace/onnx/*.onnx`、`/workspace/engines/*.plan`、`/workspace/engines/calib_*.cache` | 改为相对项目根的相对路径，或接受 `--onnx-dir` / `--engine-dir` CLI |
| [pipeline/infer_pipeline.cpp](../../pipeline/infer_pipeline.cpp) | 默认 engine 路径 `/workspace/engines/*_fp16.plan` | 同上，CLI 已经有 `--pfn-engine` / `--bb-engine` / `--e2e-engine`，**默认值改成相对路径**或环境变量 |
| [scripts/eval_mini.py](../eval_mini.py) | `/home/uceeanz/OpenPCDet/tools` 作 cwd、`../../TensorRT/results/...` 相对跳跃 | 重写为 argparse 接收 `--data-root` / `--results-json` / `--cfg` |
| [scripts/eval_nuscenes.py](../eval_nuscenes.py) | `PYTHONPATH=/workspace/deps`（容器内不存在该目录） | 删 `sys.path.insert(/workspace/deps)`；改为依赖容器自带 `nuscenes-devkit`（如未装则 `pip install nuscenes-devkit` 进 venv） |
| [scripts/verify_engines.py](../verify_engines.py) | `sys.path.insert(0, '/workspace/deps')`、`PYTHONPATH=/workspace/deps` | 同上 |
| [scripts/export_pfn.py](../export_pfn.py)、[export_bev_backbone.py](../export_bev_backbone.py) | `sys.path.insert(0, parents[2] / 'OpenPCDet')` | **保留**（导出脚本本就只在有 OpenPCDet 的环境跑，归属 02b 阶段），但加 argparse 让 OpenPCDet 路径可配 |
| [scripts/stitch_onnx.py](../stitch_onnx.py) | `PYTHONPATH=/workspace/deps` 在 docstring 里 | docstring 改写，代码本身不依赖 deps |
| [CMakeLists.txt](../../CMakeLists.txt) | `CMAKE_CUDA_ARCHITECTURES=80` 写死、TRT 路径 cache 默认值 | A100 上 80 正确，**保持**；但加注释说明，并验证 TRT 10 头文件在 `/usr/include/x86_64-linux-gnu/NvInfer.h` 实际存在 |
| 任何 `# /home/uceeanz/...` 路径注释 | docstring 里仍会引用老路径 | grep `uceeanz` 全删 |

### 推荐统一约定（审计 session 落地这一套）

- **项目根定位**：所有可执行脚本（C++ binary 也算）通过环境变量 `CENTERPOINT_ROOT` 找项目根，**未设时退化为可执行文件所在目录的上溯**（不要写死 `/workspace/...`）。
- **数据根定位**：环境变量 `NUSCENES_ROOT`，默认 `/data/sidney/datasets/nuscenes`。
- **ONNX / engine / 结果目录**：
  - ONNX：`$CENTERPOINT_ROOT/onnx/`
  - Engines：`$CENTERPOINT_ROOT/engines/`（注意 `.gitignore` 已忽略，文件本身不上 git，没问题）
  - 校准 cache：`$CENTERPOINT_ROOT/engines/calib_*.cache`
  - benchmark 结果：`$CENTERPOINT_ROOT/scripts/execute-benchmark/results/`
- **logs / markers**：`$CENTERPOINT_ROOT/scripts/execute-benchmark/{logs,markers}/`

### Phase 0 验收

1. `grep -rn '/home/uceeanz\|/workspace/deps' src include pipeline scripts cuda plugin CMakeLists.txt` 无任何命中。
2. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` 在容器内一把过。
3. 用现有 6 个 .plan（如果还在 `/workspace/engines/`）跑一次 `build/infer_pipeline --mode 4b --cuda-pillarize --cuda-postprocess --e2e-engine $CENTERPOINT_ROOT/engines/centerpoint_e2e_fp16.plan <one_sample.bin>` 能出 JSON。
4. 提交一个 commit message：`refactor: parameterize hardcoded paths via env vars`，**不**改任何业务逻辑。

---

## Phase 1 — 脚本写作（3 个并行 Claude Code session）

### 拆分原则

按"数据子系统 / 构建+评测子系统 / FP32 baseline + 汇总"切。每个 session 独占一组文件，公共 `lib/common.sh` 由 Session A 写、其他两 session 只读引用（用 `source lib/common.sh`）。**3 个 session 之间不修改对方的文件，只通过 marker 目录 + results 目录 + lib/common.sh 接口交互。**

| Session | 拥有文件 | 不准碰的文件 |
|---|---|---|
| A — 数据 | `00_download_nuscenes.sh`、`00b_export_sweeps_cache.sh`、`lib/common.sh`、`lib/_paths.sh`、`urls.txt.example`、`README.md`、对 `export_mini_sweeps.py` 的 try/except 改动（~5 行） | 其他 |
| B — 构建+评测 | `01_build_all_engines.sh`、`02_eval_grid.sh`、`lib/gpu_profile.sh`、`lib/eval_one_config.sh` | 其他 |
| C — FP32 + 汇总 | `02b_eval_openpcdet_fp32.sh`、`lib/setup_openpcdet_env.sh`、`aggregate_results.py`、`filter_val_infos.py` | 其他 |

### 共享约定（三个 session 都要遵守）

- **目录结构（开工前由 Session A 在 lib/common.sh 里 `init_dirs()` 创建）**：
  ```
  scripts/execute-benchmark/
  ├── markers/               # .downloaded_<file>, .built_<engine>, .eval_<profile>_<config> ...
  ├── logs/                  # build_*.log, eval_*.log, download_*.log
  ├── results/
  │   ├── a100_full/<config>/{raw_dets.json, bench.json, eval.json, run.log}
  │   ├── mig_1g10gb/<config>/...
  │   └── openpcdet_fp32/{eval.json, run.log}
  └── cache/                 # 小文件，比如 ONNX dump tmp；sweep 缓存走 /data/sidney/datasets/...
  ```
- **Marker 文件**：空文件即可，文件名编码状态。例如 `.built_pfn_int8`、`.eval_a100_full_4b_int8`。脚本启动时 `[[ -f marker ]] && return 0`。
- **lib/common.sh 必须提供**（Session A 负责实现，签名固定）：
  - `init_dirs()` — `mkdir -p markers logs results cache`
  - `log_info "msg"` / `log_warn "msg"` / `log_err "msg"` — 带时间戳的 stderr 输出
  - `with_retry <n> <cmd...>` — 失败重试
  - `marker_path <name>` — 输出 marker 绝对路径
  - 导出 `REPO_ROOT`、`BENCH_ROOT`、`NUSCENES_ROOT`、`SWEEPS_CACHE_DIR` 变量
- **错误处理**：所有 `.sh` 使用 `set -euo pipefail`。单个组件失败只影响自己（写 `logs/`、不写 marker、继续下一项）。
- **GPU 礼让**：`lib/gpu_profile.sh::apply_gpu_profile a100_full` 内部应先 `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader` 查空闲；若 < 60GB 自由内存，`log_warn` 并提示用户确认。
- **不要 sudo**：容器内已是 root；MIG 切换在 host 上需 sudo，但容器内 `nvidia-smi mig` 也是 root，**先验证容器是否有 MIG 权限**（多数情况共享实例下没有，要 graceful skip）。

---

### Session A 提示词（**数据子系统**）—— 复制粘贴到新 Claude Code 会话

```
你是 CenterPoint TRT 项目 benchmark 自动化套件的 Session A，负责"数据子系统"。

【开工前必读】（按顺序，全部读完再动手）
1. /workspace/CLAUDE.md
2. /workspace/demo/TensorRT-of-Pillarbased-CentrePoint/CLAUDE.md
3. /workspace/demo/TensorRT-of-Pillarbased-CentrePoint/scripts/execute-benchmark/instruction.md 全文

【你的工作目录】
cd /workspace/demo/TensorRT-of-Pillarbased-CentrePoint

【你独占的文件】（其他 session 不会动这些）
- scripts/execute-benchmark/00_download_nuscenes.sh
- scripts/execute-benchmark/00b_export_sweeps_cache.sh
- scripts/execute-benchmark/lib/common.sh        ← 三个 session 共用，但只有你写
- scripts/execute-benchmark/lib/_paths.sh        ← REPO_ROOT/NUSCENES_ROOT 等环境定义
- scripts/execute-benchmark/urls.txt.example
- scripts/execute-benchmark/README.md            ← 套件总入口文档
- 对 scripts/export_mini_sweeps.py 的 ~5 行 try/except 改动（见 instruction.md）

【你的任务】
1. 写 lib/_paths.sh：
   - 导出 REPO_ROOT（项目根，绝对路径）、BENCH_ROOT（本目录）、NUSCENES_ROOT（默认 /data/sidney/datasets/nuscenes）、SWEEPS_CACHE_DIR（默认 /data/sidney/datasets/nuscenes_sweeps_cache）
   - 允许调用方通过同名环境变量预先覆盖
2. 写 lib/common.sh：实现 instruction.md "共享约定" 里列出的全部 API（init_dirs / log_info / log_warn / log_err / with_retry / marker_path）。先 `source _paths.sh`。
3. 写 00_download_nuscenes.sh：
   - 从 urls.txt 读全量 nuScenes v1.0-trainval（10 个 blobs part + meta，共 11 行）+ mini（2 行）
   - aria2c -c -x 8 -s 8 --max-tries=3 下到 $NUSCENES_ROOT/
   - 每个文件 sha256 可选校验
   - 解压到 $NUSCENES_ROOT/v1.0-trainval/ 和 $NUSCENES_ROOT/v1.0-mini/
   - 全程 marker 化，可断点续跑
   - 下载完毕打印 `du -sh $NUSCENES_ROOT/*` 和最终目录结构
4. 写 00b_export_sweeps_cache.sh：
   - 调用 scripts/export_mini_sweeps.py，--version v1.0-trainval --split val
   - --data-root $NUSCENES_ROOT --out-dir $SWEEPS_CACHE_DIR --list-path $SWEEPS_CACHE_DIR/files.txt
   - **注意全量 nuScenes 不会有缺 sweep 的问题**，但 try/except 仍保留作为防御
   - 同时再跑一次 --version v1.0-mini --split val，输出到 $SWEEPS_CACHE_DIR/mini_val/，给 INT8 校准用
   - 抽 1 个 .bin reshape(-1,5)，col 4 应有多个不同 time_lag 值
5. 改 scripts/export_mini_sweeps.py：在 main 循环里包 try/except FileNotFoundError，跳过缺文件的 sample，~5 行。
6. 写 urls.txt.example：列全 13 个 nuScenes URL 占位（trainval meta + 10 blobs + mini meta + mini blob），格式 `<url> <filename> [<sha256>]`，加注释说明 presigned URL 1-7 天过期。
7. 写 README.md：套件总入口。包含目录结构、一键执行流程、marker 机制说明、常见错误排查。

【时间估算需写进 README】
- 00 下载：全量 trainval ~340GB，aria2 8 线程 ~3-4h
- 00b sweep cache：~6000 帧 × 200ms ≈ 25-35 min，磁盘 ~36GB

【验收】
- bash 语法干净：`bash -n *.sh && bash -n lib/*.sh`
- 干跑（不真下数据）：构造 urls.txt 只放 1 个小文件，跑一遍后 Ctrl+C 重跑应跳过
- export_mini_sweeps.py 改动后不破坏现有 mini split 行为
- 完成后给我一段 50 字内的状态汇报：哪些文件写了、哪些坑遇到了

【不准做】
- 不要碰 Session B/C 的文件（见 instruction.md 拆分表）
- 不要把数据下到 /workspace（193G 小盘会爆）

【pip 策略】
- 套件统一**直接用容器自带 python**，不建 venv。
- export_mini_sweeps.py 需要 nuscenes-devkit + pyquaternion + numpy；容器 NGC 25.02 自带 numpy，但 nuscenes-devkit / pyquaternion **多半没有**。00b 脚本开头先 `python -c "import nuscenes, pyquaternion" || pip install nuscenes-devkit pyquaternion`，装到容器层即可（ephemeral，但不影响 benchmark 跑通）。
```

---

### Session B 提示词（**构建+评测子系统**）—— 复制粘贴到新 Claude Code 会话

```
你是 CenterPoint TRT 项目 benchmark 自动化套件的 Session B，负责"engine 构建 + TRT 评测矩阵"。

【开工前必读】（按顺序）
1. /workspace/CLAUDE.md
2. /workspace/demo/TensorRT-of-Pillarbased-CentrePoint/CLAUDE.md
3. /workspace/demo/TensorRT-of-Pillarbased-CentrePoint/scripts/execute-benchmark/instruction.md 全文

【你的工作目录】
cd /workspace/demo/TensorRT-of-Pillarbased-CentrePoint

【你独占的文件】
- scripts/execute-benchmark/01_build_all_engines.sh
- scripts/execute-benchmark/02_eval_grid.sh
- scripts/execute-benchmark/lib/gpu_profile.sh
- scripts/execute-benchmark/lib/eval_one_config.sh

【你依赖但不修改的】
- scripts/execute-benchmark/lib/common.sh、lib/_paths.sh（由 Session A 写，你 source 之后用 log_info / marker_path / REPO_ROOT / NUSCENES_ROOT / SWEEPS_CACHE_DIR / BENCH_ROOT）
- src/main.cpp、src/engine_builder.cpp、src/calibrator.cpp、pipeline/infer_pipeline.cpp（Phase 0 已参数化）
- scripts/verify_engines.py、scripts/eval_nuscenes.py（Phase 0 已修复路径）

【你的任务】

1. 写 01_build_all_engines.sh：
   a. INT8 校准 dump（marker .calib_dump_done）：
      ./build/infer_pipeline --mode 4b --cuda-pillarize --cuda-postprocess --pinned-points \
        --e2e-engine ... \
        --dump-calib-dir $REPO_ROOT/engines/calib_data/ --calib-frames 500 \
        --file-list $SWEEPS_CACHE_DIR/mini_val/files.txt
      注意：dump 需要先有一个 e2e engine 才能跑，但 INT8 build 又依赖 dump → 先用 FP16 e2e dump，再 build INT8。
   b. 6 个 engine 顺序 build，每个完成立刻 verify_engines.py：
      | # | engine          | 命令                                       |
      |---|-----------------|--------------------------------------------|
      | 1 | pfn_fp16        | build/build_engine（无参，默认 FP16 三件套） |
      | 2 | backbone_head_fp16 | （build_engine 已包含）                   |
      | 3 | centerpoint_e2e_fp16 | （build_engine 已包含）                |
      | 4 | pfn_int8        | build/build_engine --int8-pfn engines/calib_data |
      | 5 | backbone_head_int8 | build/build_engine --int8-bb engines/calib_data |
      | 6 | centerpoint_e2e_int8 | build/build_engine --int8-e2e engines/calib_data |
      每个 marker：.built_pfn_fp16、.built_pfn_int8 等。
   c. verify_engines.py 失败保留 logs/build_<name>.log 但不阻塞下一项。

2. 写 lib/gpu_profile.sh：
   - apply_gpu_profile a100_full → 无操作，返回 0
   - apply_gpu_profile mig_1g10gb：
     - 先 `nvidia-smi mig -lgip` 看是否支持 MIG，不支持 return 1（让 02 graceful skip）
     - `nvidia-smi -mig 1`、`nvidia-smi mig -cgi 19 -C`、抓 MIG-<UUID> 导出 CUDA_VISIBLE_DEVICES
     - 任一步失败 return 1
   - restore_gpu_profile mig_1g10gb：`nvidia-smi mig -dci -dgi -mig 0`
   - 注意：当前容器是共享 GPU，**多数情况下没有 MIG 权限**，graceful skip 是正常路径。

3. 写 lib/eval_one_config.sh：
   - 接收 $1=config (4a_fp16|4a_int8|4b_fp16|4b_int8)、$2=gpu_profile
   - 在 results/$gpu_profile/$config/ 下：
     - 调 infer_pipeline 拿 raw_dets.json + bench.json（用 SWEEPS_CACHE_DIR/files.txt 作 --file-list；warmup 20、bench-iters 1200）
     - 调 scripts/eval_nuscenes.py 拿 eval.json（--data-root $NUSCENES_ROOT/v1.0-trainval --split val）
   - 失败保留 run.log，不写 marker
   - 4a 模式需要 --pfn-engine + --bb-engine 两路；4b 模式需要 --e2e-engine
   - 全部 4 配置共用 --cuda-pillarize --cuda-postprocess

4. 写 02_eval_grid.sh：
   - 主循环：for gpu_profile in a100_full mig_1g10gb; for config in 4a_fp16 4a_int8 4b_fp16 4b_int8
   - 每个组合先看 marker（.eval_<profile>_<config>），存在则 skip
   - 调 apply_gpu_profile → eval_one_config → 成功写 marker → restore_gpu_profile
   - apply_gpu_profile 返回非 0 时把该 profile 下所有 config 标记为 skipped 并 log_warn

【关键 CLI flag 索引（infer_pipeline.cpp 已确认存在）】
- --mode {4a,4b}
- --pfn-engine / --bb-engine / --e2e-engine
- --cuda-pillarize / --cuda-postprocess / --pinned-points / --double-buffer
- --benchmark --warmup N --bench-iters N --bench-output X.json
- --json out.json
- --file-list files.txt
- --dump-calib-dir <dir> --calib-frames N

【时间估算】
- 01 build：FP16 三件套 ~10 min，INT8 三件套 ~30-65 min（含 dump 5 min），合计 45-90 min
- 02 eval：6000 帧 × ~50ms × 8 组合 ≈ 40-50 min（不含 MIG skip 情况）

【验收】
- bash -n 全过
- 干跑：只指定一个 config 单跑通 4b_fp16 × a100_full 一格
- 完成后给我 50 字内汇报

【不准做】
- 不改 Session A/C 的文件
- 不改 src/ pipeline/ scripts/*.py 业务逻辑（Phase 0 已经做过参数化了）
- MIG 权限缺失不要硬切，按 graceful skip 设计
```

---

### Session C 提示词（**FP32 baseline + 汇总**）—— 复制粘贴到新 Claude Code 会话

```
你是 CenterPoint TRT 项目 benchmark 自动化套件的 Session C，负责"OpenPCDet PyTorch FP32 baseline + 结果汇总"。

【开工前必读】（按顺序）
1. /workspace/CLAUDE.md
2. /workspace/demo/TensorRT-of-Pillarbased-CentrePoint/CLAUDE.md
3. /workspace/demo/TensorRT-of-Pillarbased-CentrePoint/scripts/execute-benchmark/instruction.md 全文

【你的工作目录】
cd /workspace/demo/TensorRT-of-Pillarbased-CentrePoint

【你独占的文件】
- scripts/execute-benchmark/02b_eval_openpcdet_fp32.sh
- scripts/execute-benchmark/lib/setup_openpcdet_env.sh
- scripts/execute-benchmark/aggregate_results.py
- scripts/execute-benchmark/filter_val_infos.py

【你依赖但不修改的】
- lib/common.sh、lib/_paths.sh（Session A 写）
- results/$gpu_profile/$config/eval.json + bench.json（Session B 写出，schema 在下面定义）
- scripts/export_*.py、scripts/eval_*.py（Phase 0 已参数化）

【关键约束（与第一版的最大区别）】
- 当前容器是 NGC pytorch 25.02，**直接用容器自带 python**，不建 venv（用户已确认）。pip 包装到 `/usr/local/lib/python3.*/site-packages/`，容器重建会丢；setup_openpcdet_env.sh 用 marker 跳过重复安装，重建后重跑会自动再装。
- OpenPCDet 仓库不在项目里，必须 git clone 一份。位置：`$REPO_ROOT/external/OpenPCDet`。**注意 .gitignore 已忽略 deps/，但 external/ 没忽略——你需要顺手在 .gitignore 加一行 `external/`。**
- ckpt：`centerpoint_pillar_nuscenes.pth` 从 OpenPCDet model zoo 下到 `external/OpenPCDet/ckpts/`。setup 脚本检查存在则跳过，否则 wget。

【你的任务】

1. 写 lib/setup_openpcdet_env.sh：
   - 检查 marker .openpcdet_env_ready，存在则 no-op
   - **直接用容器自带 python**（`/usr/bin/python` / `/usr/local/bin/python`，NGC 25.02 自带 torch + CUDA 12.8），**不建 venv**
   - pip install 顺序：
     - `pip install spconv-cu120`（NGC 25.02 是 cu128，spconv-cu120 一般可加载；不行 fallback `spconv-cu118` → 都不行 log_err 让 Sidney 介入）
     - `pip install nuscenes-devkit pyquaternion`（如果 Session A 装过会幂等跳过）
     - `cd external/OpenPCDet && pip install -e .`
   - git clone https://github.com/open-mmlab/OpenPCDet.git 到 external/OpenPCDet（已存在则 skip）
   - 下 ckpt：wget OpenPCDet model zoo 的 centerpoint_pillar_nuscenes.pth 到 external/OpenPCDet/ckpts/（model zoo URL 在 OpenPCDet README）
   - 验证 `python -c "import pcdet; import spconv; import nuscenes"` 全部 import 成功后写 marker
   - **重要踩坑**：NGC 容器自带的 torch 是源码编译版，pip install OpenPCDet 时 `pip install -e .` 可能会触发依赖解析想升级 torch → 加 `--no-deps` 装 pcdet 自身，再手动 `pip install easydict tqdm tensorboardX` 等次级依赖；防止 pip 把容器内的 torch 拆了。
   - 容器若被 `docker rm` 重建过，marker 还在但 site-packages 没了 → 脚本开头先 `python -c "import pcdet"` 探测，import 失败就删 marker 重装。

2. 写 filter_val_infos.py（~30 行）：
   - 输入：OpenPCDet 生成的 nuscenes_infos_10sweeps_val.pkl + SWEEPS_CACHE_DIR/files.txt
   - 按 lidar 文件名 / sample token 过滤 infos pkl，输出 nuscenes_infos_10sweeps_val_subset.pkl
   - 用途：避免 OpenPCDet dataloader 读到不存在的 sweep（虽然全量数据没这问题，但保留防御）

3. 写 02b_eval_openpcdet_fp32.sh：
   - marker .openpcdet_fp32_done 跳过
   - source lib/setup_openpcdet_env.sh
   - cd external/OpenPCDet && 生成 infos pkl：
     python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
       --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
     （注意：cfg 内 DATA_PATH 要指向 $NUSCENES_ROOT，可能需要写一份本地 override yaml）
   - 跑 filter_val_infos.py
   - 跑 tools/test.py：
     python tools/test.py \
       --cfg_file tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml \
       --batch_size 1 --ckpt ckpts/centerpoint_pillar_nuscenes.pth \
       --eval_tag fp32_baseline_subset
   - 把输出 result.pkl 转成 TRT 兼容 eval.json 格式（schema 见下），落 results/openpcdet_fp32/eval.json
   - 写 marker

4. 写 aggregate_results.py：
   - 扫 results/{a100_full,mig_1g10gb}/{4a,4b}_{fp16,int8}/ + results/openpcdet_fp32/
   - 读每个目录下的 eval.json + bench.json（FP32 baseline 只有 eval.json）
   - 输出：
     - benchmark_report.csv（列：gpu_profile, config, mAP, NDS, mATE, mAOE, p50_ms, p99_ms, max_ms, fps）
     - benchmark_report.md（同上 + Markdown 表 + 可选 matplotlib mAP-FPS 散点图）
   - FP32 baseline 单独一行，FPS / 延迟列填 "N/A"
   - 缺失组合（比如 MIG skip）也要在表里出现，标 "skipped"

【需要约定 schema】
eval.json（Session B 的 eval_nuscenes.py 输出和你这边 OpenPCDet 转出的格式必须一致）：
{
  "mAP": float, "NDS": float,
  "per_class_AP": {"car": float, ...},
  "TP_metrics": {"mATE": float, "mASE": float, "mAOE": float, "mAVE": float, "mAAE": float}
}
bench.json（Session B 的 infer_pipeline --bench-output 已经生成，你只读不改）：
{
  "frames": int,
  "p50_ms": float, "p99_ms": float, "max_ms": float, "mean_ms": float,
  "fps": float,
  "per_stage": {"h2d": {...}, "pillarize": {...}, "infer": {...}, "decode": {...}}
}
若实际 infer_pipeline 输出字段不同，以 infer_pipeline.cpp 实际为准，aggregate_results.py 兼容即可。

【时间估算】
- setup（git clone OpenPCDet + pip install spconv/pcdet + 下 ckpt）：首次 ~10-20 min（spconv 走预编译 wheel，比第一版的 venv 路径快）
- OpenPCDet 推理 6000 帧 ~150ms ≈ 15-20 min
- nuScenes 官方 eval ~5 min
- 总计首次 30-45 min，重跑（marker 命中）20-25 min

【验收】
- bash -n + python -m py_compile 全过
- 干跑 aggregate_results.py（用伪数据 results/ 结构）能出 CSV/MD
- 完成后给我 50 字内汇报

【不准做】
- 不改 Session A/B 文件
- 不下任何数据到 /workspace 大文件区（OpenPCDet repo 代码本身 ~50MB 可以；ckpt ~150MB 放 external/OpenPCDet/ckpts/ 也可以；val 评测产物中 result.pkl < 100MB 也行）
- 不假设 /home/uceeanz/... 或 /workspace/deps 存在（这是老 plan 残留）
- 不要建 venv（用户已确认直接用容器 python）
```

---

## Phase 2 — Brev 执行流程（脚本写完后）

```bash
# 进入项目根
cd /workspace/demo/TensorRT-of-Pillarbased-CentrePoint
cd scripts/execute-benchmark

# 1. 拷贝 nuScenes URL（从 nuscenes.org 登录后导出 13 个）
cp urls.txt.example urls.txt && vim urls.txt

# 2. 一键跑（marker 机制保证断点续跑）
bash 00_download_nuscenes.sh         # 3-4 h（一次性）
bash 00b_export_sweeps_cache.sh      # 25-35 min（一次性）
bash 01_build_all_engines.sh         # 45-90 min（INT8 是大头）
bash 02_eval_grid.sh                 # 40-50 min（MIG 可能 skip）
bash 02b_eval_openpcdet_fp32.sh      # 首次 35-65 min，重跑 20-25 min

# 3. 汇总
python aggregate_results.py

# 4. 看报告
cat benchmark_report.md
```

**总计：5-8h 无人值守。** 首次比第一版长很多，主要是全量 trainval 下载 + 推理帧数从 2400 涨到 6000。

---

## Phase 3 — 验证方法（脚本写完后局部小跑，别等到 Brev 上才发现 bug）

| 步骤 | 验证 | 通过条件 |
|---|---|---|
| Phase 0 | grep 全项目无 `uceeanz`/`/workspace/deps` | 命中数 = 0 |
| Phase 0 | cmake + 跑一个旧 sample.bin | infer_pipeline 出 JSON |
| Session A 干跑 | 用 1 个小文件的 urls.txt.test，Ctrl+C 重跑 | 跳过已下载项 |
| Session A 干跑 | mini --split val 跑 sweep export | 输出 ~81 帧，col 4 多值 |
| Session B 干跑 | 单跑 4b_fp16 × a100_full | results 目录结构 + JSON 完整 |
| Session B 干跑 | apply_gpu_profile mig_1g10gb（容器无权限） | 返回 1 + log_warn，主循环 graceful skip |
| Session C 干跑 | aggregate_results.py 喂伪造 results | 出 CSV + MD |
| Session C 干跑 | 02b 用 mini val 81 帧 | OpenPCDet test 出 result.pkl → 转 eval.json |

---

## 风险与回退

| 风险 | 概率 | 应对 |
|---|---|---|
| nuScenes presigned URL 过期（1-7 天） | 中 | 00 脚本检测 403 中断，提示重拷 urls.txt |
| /data/sidney/datasets 空间不足（340GB） | 低 | 跑 00 前 `df -h /data` 检查，预留 500GB |
| GPU 被团队占用 | 中-高 | lib/common.sh 在 init 时 `nvidia-smi`；evaluator 加 `CUDA_VISIBLE_DEVICES` 显式选卡 |
| 共享容器无 MIG 权限 | 高 | gpu_profile.sh graceful skip，留 a100_full 一档 |
| spconv-cu120 与 NGC 25.02 (cu128) 不兼容 | 中 | fallback 链 cu120 → cu118 → 报错；最坏跳过 02b（FP16 当上限） |
| INT8 校准 OOM | 低 | 单 engine 失败不影响其他；调小 calib batch |
| TRT 10 API 与老代码（IPluginV3）不兼容 | 低 | 项目已经在用 IPluginV3（见 plugin/scatter_plugin.cpp），TRT 10 原生支持 |
| 容器层 pip 包被 docker rm 清掉 | 中 | 接受 ephemeral，02b 的 setup 脚本 marker + `import pcdet` 探测：探测失败自动删 marker 重装。**不**建 venv（用户决策） |
| pip install -e OpenPCDet 把容器自带 torch 拆了 | 中-高 | 用 `pip install --no-deps -e .` 装 pcdet 本体，次级依赖手动按需补，保护 NGC 自带 torch |
| Phase 0 改动破坏现有 binary 行为 | 中 | Phase 0 严禁改业务逻辑；改完跑一个 sample.bin 回归 |

---

## 关键文件清单

**新建（13 个）**：
- 上述 3 个 session 各自的文件，共 13 个（含 README.md、urls.txt.example）

**Phase 0 改造（不限于此，审计 session 自行 grep 兜底）**：
- src/main.cpp、pipeline/infer_pipeline.cpp（默认路径参数化）
- scripts/eval_mini.py、eval_nuscenes.py、verify_engines.py、stitch_onnx.py、export_*.py（删 `/workspace/deps` / `/home/uceeanz`）
- scripts/export_mini_sweeps.py（try/except FileNotFoundError，~5 行，归 Session A 改）

**复用不动**：
- src/engine_builder.cpp、src/calibrator.cpp、include/*.h、cuda/*.cu、plugin/scatter_plugin.cpp、CMakeLists.txt
