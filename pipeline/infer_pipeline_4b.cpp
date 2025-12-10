// 4b 端到端推理 pipeline：单引擎（含 PillarScatter plugin），不再 host 端调 scatter。
//
// 与 4a (pipeline/infer_pipeline.cpp) 区别仅在 runOneFrame：
//   4a：pfn_ctx.infer() → cudaMemset + launchPillarScatter → bb_ctx.infer()
//   4b：e2e_ctx.infer()  // scatter 已封装进图内
// 其余（数据加载、pillarize、post-process、benchmark）逻辑完全一致，
// 故选择拷贝而非抽象出共用模块——便于直接对比阅读。
//
// 支持 --cuda-pillarize / --cuda-postprocess：CUDA pillarize 直接写到 e2e 引擎的
// input device buffer；CUDA postprocess 直接读 e2e 引擎的 output device buffer，
// 全程零中间 D2H/H2D（除点云 H2D 与候选 D2H）。

#include "infer_context.h"
#include "scatter_plugin.h"
#include "pillarize.h"
#include "pillarize_cuda.h"
#include "postprocess.h"
#include "postprocess_cuda.h"
#include "bench.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <vector>

static constexpr float X_MIN = -51.2f, X_MAX = 51.2f;
static constexpr float Y_MIN = -51.2f, Y_MAX = 51.2f;
static constexpr float Z_MIN = -5.0f,  Z_MAX = 3.0f;
static constexpr float VX = 0.2f, VY = 0.2f, VZ = 8.0f;
static constexpr int   MAX_PTS     = 20;
static constexpr int   MAX_PILLARS = 30000;
static constexpr int   C_IN        = 11;
static constexpr int   BEV_H = 512, BEV_W = 512;
static constexpr int   OUT_STRIDE  = 4;
static constexpr float SCORE_THRESH = 0.1f;

static const int   TASK_NUM_CLS[6]    = {1, 2, 2, 1, 2, 2};
static const float TASK_NMS_RADIUS[6] = {2.0f, 4.0f, 6.0f, 0.5f, 1.5f, 0.5f};
static constexpr int MAX_CANDS_PER_TASK = 4096;

static const char* CLASS_NAMES[10] = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
};

static std::vector<float> loadBin(const std::string& path, int& N) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("无法打开: " + path);
    size_t bytes = f.tellg();
    f.seekg(0);
    N = (int)(bytes / (5 * sizeof(float)));
    std::vector<float> pts(N * 5);
    f.read(reinterpret_cast<char*>(pts.data()), bytes);
    return pts;
}

// 直接读到一个预分配的 pinned host buffer 里。容量不够时按需 cudaMallocHost 扩容。
// 用于 --pinned-points 路径，省掉 std::vector 中转和 cudaMemcpyAsync 内部的 bounce 拷贝。
static int loadBinIntoPinned(const std::string& path,
                             float*& pinned, size_t& pinned_capacity_bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("无法打开: " + path);
    size_t bytes = f.tellg();
    f.seekg(0);
    if (bytes > pinned_capacity_bytes) {
        if (pinned) cudaFreeHost(pinned);
        pinned_capacity_bytes = bytes + (bytes >> 2);  // 25% slack 减少抖动期重分配
        cudaHostAlloc(&pinned, pinned_capacity_bytes, cudaHostAllocDefault);
    }
    f.read(reinterpret_cast<char*>(pinned), bytes);
    return (int)(bytes / (5 * sizeof(float)));
}

static void writeFrameJson(std::ostream& os,
                           const std::string& key,
                           const std::vector<Detection>& dets) {
    os << std::fixed << std::setprecision(4);
    os << "  \"" << key << "\": [\n";
    for (size_t i = 0; i < dets.size(); i++) {
        const auto& d = dets[i];
        os << "    {\"class\": \"" << CLASS_NAMES[d.cls_id] << "\""
           << ", \"score\": " << d.score
           << ", \"x\": " << d.x << ", \"y\": " << d.y << ", \"z\": " << d.z
           << ", \"w\": " << d.w << ", \"l\": " << d.l << ", \"h\": " << d.h
           << ", \"rot\": " << d.rot
           << ", \"vx\": " << d.vx << ", \"vy\": " << d.vy
           << "}";
        if (i + 1 < dets.size()) os << ",";
        os << "\n";
    }
    os << "  ]";
}

int main(int argc, char** argv) {
    centerpoint_trt::registerPillarScatterPlugin();

    std::string json_path;
    std::string file_list_path;
    std::string engine_path = "/workspace/engines/centerpoint_e2e_fp16.plan";
    std::vector<std::string> bin_files;

    bool benchmark = false;
    int  warmup_iters = 10;
    int  bench_iters = 200;
    std::string bench_output;
    bool cuda_pillarize = false;
    bool cuda_postprocess = false;
    bool pinned_points = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc)              json_path = argv[++i];
        else if (arg == "--file-list" && i + 1 < argc)    file_list_path = argv[++i];
        else if (arg == "--engine" && i + 1 < argc)       engine_path = argv[++i];
        else if (arg == "--benchmark")                    benchmark = true;
        else if (arg == "--warmup" && i + 1 < argc)       warmup_iters = std::stoi(argv[++i]);
        else if (arg == "--bench-iters" && i + 1 < argc)  bench_iters = std::stoi(argv[++i]);
        else if (arg == "--bench-output" && i + 1 < argc) bench_output = argv[++i];
        else if (arg == "--cuda-pillarize")               cuda_pillarize = true;
        else if (arg == "--cuda-postprocess")             cuda_postprocess = true;
        else if (arg == "--pinned-points")                pinned_points = true;
        else                                              bin_files.push_back(arg);
    }

    if (!file_list_path.empty()) {
        std::ifstream lf(file_list_path);
        if (!lf) throw std::runtime_error("无法打开 file list: " + file_list_path);
        std::string line;
        while (std::getline(lf, line)) if (!line.empty()) bin_files.push_back(line);
    }

    if (bin_files.empty()) {
        std::cerr << "Usage: infer_pipeline_4b <file.bin> [...] [--file-list files.txt]\n"
                  << "                          [--engine path/to/engine.plan]\n"
                  << "                          [--json out.json]\n"
                  << "                          [--cuda-pillarize] [--cuda-postprocess] [--pinned-points]\n"
                  << "                          [--benchmark [--warmup N] [--bench-iters N] [--bench-output bench.json]]\n";
        return 1;
    }

    InferContext e2e_ctx(engine_path);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // CUDA pillarize 路径需要的 workspace + 点云 device buffer + host pinned pillar 计数
    float*        d_points = nullptr;
    int           d_points_capacity = 0;
    unsigned int* d_voxel_point_count = nullptr;
    float*        d_voxels_dense = nullptr;
    float*        d_voxel_features_compact = nullptr;
    unsigned int* d_voxel_num_points = nullptr;
    unsigned int* d_pillar_num = nullptr;
    unsigned int* h_pillar_num = nullptr;
    if (cuda_pillarize) {
        PillarizeWorkspaceBytes wb;
        pillarizeCudaWorkspaceBytes(BEV_H, BEV_W, MAX_PTS, MAX_PILLARS, wb);
        cudaMalloc(&d_voxel_point_count,      wb.voxel_point_count_bytes);
        cudaMalloc(&d_voxels_dense,           wb.voxels_dense_bytes);
        cudaMalloc(&d_voxel_features_compact, wb.voxel_features_bytes);
        cudaMalloc(&d_voxel_num_points,       wb.voxel_num_points_bytes);
        cudaMalloc(&d_pillar_num,             wb.pillar_num_bytes);
        cudaHostAlloc(&h_pillar_num, sizeof(unsigned int), cudaHostAllocDefault);
        std::cerr << "[pipeline] pillarize=CUDA  workspace="
                  << (wb.voxel_point_count_bytes + wb.voxels_dense_bytes + wb.voxel_features_bytes
                      + wb.voxel_num_points_bytes + wb.pillar_num_bytes) / (1024*1024)
                  << " MB\n";
    } else {
        std::cerr << "[pipeline] pillarize=CPU\n";
    }

    // CUDA postprocess workspace
    Detection*    d_cand_buffer        = nullptr;
    unsigned int* d_cand_count         = nullptr;
    Detection*    h_cand_buffer_pinned = nullptr;
    unsigned int* h_cand_count_pinned  = nullptr;
    if (cuda_postprocess) {
        PostprocessWorkspaceBytes pwb;
        postprocessCudaWorkspaceBytes(6, MAX_CANDS_PER_TASK, pwb);
        cudaMalloc(&d_cand_buffer,                                pwb.cand_buffer_bytes);
        cudaMalloc(&d_cand_count,                                 pwb.cand_count_bytes);
        cudaHostAlloc(&h_cand_buffer_pinned, pwb.cand_buffer_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_cand_count_pinned,  pwb.cand_count_bytes,  cudaHostAllocDefault);
        std::cerr << "[pipeline] postprocess=CUDA  workspace="
                  << (pwb.cand_buffer_bytes + pwb.cand_count_bytes) / 1024
                  << " KB\n";
    } else {
        std::cerr << "[pipeline] postprocess=CPU\n";
    }

    auto runOneFrame = [&](const float* pts_ptr, int N, StageTimer* timer)
                       -> std::vector<Detection> {
        if (timer) timer->beginFrameGpu(stream);

        int P = 0;
        if (cuda_pillarize) {
            if (timer) timer->begin("h2d_points");
            if (N > d_points_capacity) {
                if (d_points) cudaFree(d_points);
                d_points_capacity = N + (N >> 2);
                cudaMalloc(&d_points, d_points_capacity * 5 * sizeof(float));
            }
            cudaMemcpyAsync(d_points, pts_ptr, N * 5 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            if (timer) timer->end("h2d_points");

            // pillarize 直接写到 e2e 引擎的 input device buffer
            if (timer) timer->begin("pillarize");
            float* d_e2e_features = static_cast<float*>(
                e2e_ctx.getDeviceBuffer("pillar_features"));
            float* d_e2e_mask = static_cast<float*>(
                e2e_ctx.getDeviceBuffer("pillar_mask"));
            int*   d_e2e_coords = static_cast<int*>(
                e2e_ctx.getDeviceBuffer("pillar_coords"));
            launchPillarizeCuda(
                d_points, N,
                X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                VX, VY, VZ, BEV_H, BEV_W, MAX_PTS, MAX_PILLARS,
                d_voxel_point_count, d_voxels_dense, d_voxel_features_compact,
                d_voxel_num_points, d_pillar_num,
                d_e2e_features, d_e2e_mask, d_e2e_coords,
                h_pillar_num, stream);
            cudaStreamSynchronize(stream);
            P = static_cast<int>(*h_pillar_num);
            if (P > MAX_PILLARS) P = MAX_PILLARS;
            if (timer) timer->end("pillarize");

            if (timer) timer->begin("infer_e2e");
            e2e_ctx.setInputShape("pillar_features", {P, MAX_PTS, C_IN});
            e2e_ctx.setInputShape("pillar_mask",     {P, MAX_PTS});
            e2e_ctx.setInputShape("pillar_coords",   {P, 4});
            // device buffer 已被 launchPillarizeCuda 填好，跳过 setInput 的 H2D
            e2e_ctx.infer();
            if (timer) timer->end("infer_e2e");
        } else {
            if (timer) timer->begin("pillarize");
            auto batch = pillarize(pts_ptr, N,
                X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                VX, VY, VZ, MAX_PTS, MAX_PILLARS);
            P = batch.P;
            if (timer) timer->end("pillarize");

            if (timer) timer->begin("infer_e2e");
            e2e_ctx.setInputShape("pillar_features", {P, MAX_PTS, C_IN});
            e2e_ctx.setInputShape("pillar_mask",     {P, MAX_PTS});
            e2e_ctx.setInputShape("pillar_coords",   {P, 4});
            e2e_ctx.setInput("pillar_features",
                             batch.features.data(), P * MAX_PTS * C_IN * sizeof(float));
            e2e_ctx.setInput("pillar_mask",
                             batch.mask.data(),     P * MAX_PTS * sizeof(float));
            e2e_ctx.setInput("pillar_coords",
                             batch.coords.data(),   P * 4 * sizeof(int));
            e2e_ctx.infer();
            if (timer) timer->end("infer_e2e");
        }

        if (timer) timer->begin("decode_nms");
        std::vector<Detection> all_dets;

        if (cuda_postprocess) {
            cudaMemsetAsync(d_cand_count, 0, sizeof(unsigned int) * 6, stream);

            int cls_offset_gpu = 0;
            for (int t = 0; t < 6; t++) {
                std::string pre = "task" + std::to_string(t) + "_";
                int nc = TASK_NUM_CLS[t];
                auto hm_shape = e2e_ctx.getOutputShape(pre + "hm");
                int H = hm_shape[2], W = hm_shape[3];

                const float* d_hm       = static_cast<const float*>(e2e_ctx.getDeviceBuffer(pre + "hm"));
                const float* d_center   = static_cast<const float*>(e2e_ctx.getDeviceBuffer(pre + "center"));
                const float* d_center_z = static_cast<const float*>(e2e_ctx.getDeviceBuffer(pre + "center_z"));
                const float* d_dim      = static_cast<const float*>(e2e_ctx.getDeviceBuffer(pre + "dim"));
                const float* d_rot      = static_cast<const float*>(e2e_ctx.getDeviceBuffer(pre + "rot"));
                const float* d_vel      = static_cast<const float*>(e2e_ctx.getDeviceBuffer(pre + "vel"));

                launchDecodeTaskCuda(
                    d_hm, d_center, d_center_z, d_dim, d_rot, d_vel,
                    H, W, nc, t, cls_offset_gpu, MAX_CANDS_PER_TASK,
                    SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE,
                    d_cand_buffer + (size_t)t * MAX_CANDS_PER_TASK,
                    d_cand_count + t,
                    stream);
                cls_offset_gpu += nc;
            }

            cudaMemcpyAsync(h_cand_count_pinned, d_cand_count,
                            sizeof(unsigned int) * 6,
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_cand_buffer_pinned, d_cand_buffer,
                            sizeof(Detection) * 6 * MAX_CANDS_PER_TASK,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (int t = 0; t < 6; t++) {
                unsigned int K = h_cand_count_pinned[t];
                if (K > (unsigned int)MAX_CANDS_PER_TASK) {
                    std::cerr << "[warn] task " << t << " candidate overflow: "
                              << K << " > " << MAX_CANDS_PER_TASK << "\n";
                    K = MAX_CANDS_PER_TASK;
                }
                std::vector<Detection> dets(
                    h_cand_buffer_pinned + (size_t)t * MAX_CANDS_PER_TASK,
                    h_cand_buffer_pinned + (size_t)t * MAX_CANDS_PER_TASK + K);
                dets = circleNMS(dets, TASK_NMS_RADIUS[t]);
                all_dets.insert(all_dets.end(), dets.begin(), dets.end());
            }
        } else {
            int cls_offset = 0;
            for (int t = 0; t < 6; t++) {
                std::string pre = "task" + std::to_string(t) + "_";
                int nc = TASK_NUM_CLS[t];

                auto hm_shape = e2e_ctx.getOutputShape(pre + "hm");
                int H = hm_shape[2], W = hm_shape[3];
                size_t map = (size_t)(H * W) * sizeof(float);

                std::vector<float> hm(nc*H*W), center(2*H*W), center_z(H*W);
                std::vector<float> dim(3*H*W), rot(2*H*W), vel(2*H*W);

                e2e_ctx.getOutput(pre+"hm",       hm.data(),       nc*map);
                e2e_ctx.getOutput(pre+"center",   center.data(),    2*map);
                e2e_ctx.getOutput(pre+"center_z", center_z.data(),  map);
                e2e_ctx.getOutput(pre+"dim",      dim.data(),       3*map);
                e2e_ctx.getOutput(pre+"rot",      rot.data(),       2*map);
                e2e_ctx.getOutput(pre+"vel",      vel.data(),       2*map);

                auto dets = decodeTask(
                    hm.data(), center.data(), center_z.data(),
                    dim.data(), rot.data(), vel.data(),
                    H, W, nc, t, cls_offset,
                    SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE);
                dets = circleNMS(dets, TASK_NMS_RADIUS[t]);
                all_dets.insert(all_dets.end(), dets.begin(), dets.end());
                cls_offset += nc;
            }
        }
        if (timer) timer->end("decode_nms");

        if (timer) timer->endFrameGpu(stream);
        return all_dets;
    };

    if (benchmark) {
        std::cerr << "[benchmark] preloading " << bin_files.size()
                  << " bin files into "
                  << (pinned_points ? "pinned" : "pageable")
                  << " memory (excluded from timing)...\n";

        // 两路存储：pinned 或 pageable（互斥）
        std::vector<std::vector<float>> preloaded_pageable;
        std::vector<float*>             preloaded_pinned;
        std::vector<int> preloaded_N;
        preloaded_N.reserve(bin_files.size());

        size_t total_pinned_bytes = 0;
        if (pinned_points) {
            preloaded_pinned.reserve(bin_files.size());
            for (const auto& p : bin_files) {
                float* buf = nullptr;
                size_t cap = 0;
                int N = loadBinIntoPinned(p, buf, cap);
                preloaded_pinned.push_back(buf);
                preloaded_N.push_back(N);
                total_pinned_bytes += cap;
            }
            std::cerr << "[benchmark] preloaded pinned host "
                      << total_pinned_bytes / (1024*1024) << " MB\n";
        } else {
            preloaded_pageable.reserve(bin_files.size());
            for (const auto& p : bin_files) {
                int N;
                preloaded_pageable.push_back(loadBin(p, N));
                preloaded_N.push_back(N);
            }
        }
        std::cerr << "[benchmark] warmup=" << warmup_iters
                  << "  iters=" << bench_iters << "\n";

        auto getPts = [&](int idx) -> const float* {
            return pinned_points ? preloaded_pinned[idx]
                                 : preloaded_pageable[idx].data();
        };

        StageTimer timer;
        int n_files = (int)preloaded_N.size();
        for (int i = 0; i < warmup_iters; i++) {
            int idx = i % n_files;
            (void)runOneFrame(getPts(idx), preloaded_N[idx], nullptr);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < bench_iters; i++) {
            int idx = i % n_files;
            (void)runOneFrame(getPts(idx), preloaded_N[idx], &timer);
            timer.commitFrame();
        }

        timer.summary(std::cout);
        if (!bench_output.empty()) {
            timer.writeJson(bench_output);
            std::cerr << "[benchmark] wrote " << bench_output << "\n";
        }

        if (pinned_points) {
            for (auto* p : preloaded_pinned) cudaFreeHost(p);
        }
    } else {
        std::ofstream jf;
        if (!json_path.empty()) {
            jf.open(json_path);
            jf << "{\n";
        }

        // 单帧路径用一个共享 pinned scratch buffer，按需扩容（首帧分配后稳态零开销）
        float* h_pinned_scratch = nullptr;
        size_t pinned_scratch_cap = 0;

        int total = (int)bin_files.size();
        for (int fi = 0; fi < total; fi++) {
            const auto& bin_path = bin_files[fi];

            int N;
            const float* pts_ptr;
            std::vector<float> pts_pageable;
            if (pinned_points) {
                N = loadBinIntoPinned(bin_path, h_pinned_scratch, pinned_scratch_cap);
                pts_ptr = h_pinned_scratch;
            } else {
                pts_pageable = loadBin(bin_path, N);
                pts_ptr = pts_pageable.data();
            }

            auto all_dets = runOneFrame(pts_ptr, N, nullptr);

            if (!json_path.empty()) {
                writeFrameJson(jf, bin_path, all_dets);
                if (fi + 1 < total) jf << ",";
                jf << "\n";
            } else {
                std::cout << "\n[" << (fi+1) << "/" << total << "] " << bin_path
                          << "  points=" << N << "  dets=" << all_dets.size() << "\n";
                for (auto& d : all_dets) {
                    printf("  %-25s score=%.3f  xyz=(%.1f,%.1f,%.1f)  wlh=(%.1f,%.1f,%.1f)  rot=%.2f\n",
                           CLASS_NAMES[d.cls_id], d.score,
                           d.x, d.y, d.z, d.w, d.l, d.h, d.rot);
                }
            }

            if (!(json_path.empty() || total == 1)
                && ((fi+1) % 20 == 0 || fi+1 == total)) {
                std::cerr << "  [" << (fi+1) << "/" << total << "] done\n";
            }
        }

        if (!json_path.empty()) {
            jf << "}\n";
            std::cerr << "JSON written to " << json_path << "\n";
        }

        if (h_pinned_scratch) cudaFreeHost(h_pinned_scratch);
    }

    if (cuda_pillarize) {
        if (d_points) cudaFree(d_points);
        cudaFree(d_voxel_point_count);
        cudaFree(d_voxels_dense);
        cudaFree(d_voxel_features_compact);
        cudaFree(d_voxel_num_points);
        cudaFree(d_pillar_num);
        cudaFreeHost(h_pillar_num);
    }
    if (cuda_postprocess) {
        cudaFree(d_cand_buffer);
        cudaFree(d_cand_count);
        cudaFreeHost(h_cand_buffer_pinned);
        cudaFreeHost(h_cand_count_pinned);
    }
    cudaStreamDestroy(stream);
    return 0;
}
