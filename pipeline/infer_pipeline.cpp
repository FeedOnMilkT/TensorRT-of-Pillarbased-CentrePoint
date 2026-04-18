// 统一推理 pipeline：通过 --mode {4a,4b} 在两条路径间切换。
//   4a：pfn_engine + 显式 scatter CUDA kernel + backbone+head_engine
//   4b：单一 e2e_engine（PillarScatter 已封装为 IPluginV3，scatter 在图内）
//
// 除推理中段外，其他全部共享：参数解析、CUDA pillarize/postprocess workspace、
// pinned 点云加载、INT8 校准 dump、benchmark 与单帧 JSON 输出循环。
//
// 校准 dump 在两个 mode 都可用：features/mask/coords 三路从 pillarize 写入的
// device buffer D2H（CUDA 路径）或 host batch（CPU 路径）落盘。spatial_features
// 仅 4a 有意义（4b 内部 scatter 中间结果不可见），dump 自然只在 4a 触发。

#include "infer_context.h"
#include "scatter_kernel.h"
#include "scatter_plugin.h"
#include "pillarize.h"
#include "pillarize_cuda.h"
#include "postprocess.h"
#include "postprocess_cuda.h"
#include "bench.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <nvtx3/nvToolsExt.h>


static constexpr float X_MIN = -51.2f, X_MAX = 51.2f;
static constexpr float Y_MIN = -51.2f, Y_MAX = 51.2f;
static constexpr float Z_MIN = -5.0f,  Z_MAX = 3.0f;
static constexpr float VX = 0.2f, VY = 0.2f, VZ = 8.0f;
static constexpr int   MAX_PTS     = 20;
static constexpr int   MAX_PILLARS = 30000;
static constexpr int   C_IN  = 11;
static constexpr int   C_OUT = 64;
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

// INT8 校准张量 dump：所有帧统一 pad/截断到 OPT 形状（OPT_P = 12000）。
// pad 区 mask/features/coords 全置 0；这样校准期 TRT 直接按 OPT shape 走。
static constexpr int OPT_P = 12000;

static void writeCalibBin(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法写入: " + path);
    f.write(reinterpret_cast<const char*>(data), bytes);
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

enum Mode { M4A, M4B };

struct DoubleBufferSlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t frame_start = nullptr;
    cudaEvent_t frame_stop = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t pillarize_stop = nullptr;
    cudaEvent_t infer_start = nullptr;
    cudaEvent_t infer_stop = nullptr;
    cudaEvent_t decode_start = nullptr;
    cudaEvent_t decode_stop = nullptr;

    bool in_flight = false;
    bool measured = false;
    int frame_index = -1;
    std::string key;

    float* h_points_pinned = nullptr;
    size_t h_points_cap = 0;
    float* d_points = nullptr;
    int d_points_capacity = 0;

    unsigned int* d_voxel_point_count = nullptr;
    float* d_voxels_dense = nullptr;
    float* d_voxel_features_compact = nullptr;
    unsigned int* d_voxel_num_points = nullptr;
    unsigned int* d_pillar_num = nullptr;
    unsigned int* h_pillar_num = nullptr;

    Detection* d_cand_buffer = nullptr;
    unsigned int* d_cand_count = nullptr;
    Detection* h_cand_buffer_pinned = nullptr;
    unsigned int* h_cand_count_pinned = nullptr;

    std::unique_ptr<InferContext> e2e_ctx;
};

static void initDoubleBufferSlot(DoubleBufferSlot& slot,
                                 const InferEngine& shared_engine) {
    cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming);
    cudaEventCreate(&slot.frame_start);
    cudaEventCreate(&slot.frame_stop);
    cudaEventCreate(&slot.h2d_start);
    cudaEventCreate(&slot.h2d_stop);
    cudaEventCreate(&slot.pillarize_stop);
    cudaEventCreate(&slot.infer_start);
    cudaEventCreate(&slot.infer_stop);
    cudaEventCreate(&slot.decode_start);
    cudaEventCreate(&slot.decode_stop);

    slot.e2e_ctx = std::make_unique<InferContext>(shared_engine);
    slot.e2e_ctx->setStream(slot.stream);

    PillarizeWorkspaceBytes wb;
    pillarizeCudaWorkspaceBytes(BEV_H, BEV_W, MAX_PTS, MAX_PILLARS, wb);
    cudaMalloc(&slot.d_voxel_point_count,      wb.voxel_point_count_bytes);
    cudaMalloc(&slot.d_voxels_dense,           wb.voxels_dense_bytes);
    cudaMalloc(&slot.d_voxel_features_compact, wb.voxel_features_bytes);
    cudaMalloc(&slot.d_voxel_num_points,       wb.voxel_num_points_bytes);
    cudaMalloc(&slot.d_pillar_num,             wb.pillar_num_bytes);
    cudaHostAlloc(&slot.h_pillar_num, sizeof(unsigned int), cudaHostAllocDefault);

    PostprocessWorkspaceBytes pwb;
    postprocessCudaWorkspaceBytes(6, MAX_CANDS_PER_TASK, pwb);
    cudaMalloc(&slot.d_cand_buffer, pwb.cand_buffer_bytes);
    cudaMalloc(&slot.d_cand_count,  pwb.cand_count_bytes);
    cudaHostAlloc(&slot.h_cand_buffer_pinned, pwb.cand_buffer_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&slot.h_cand_count_pinned,  pwb.cand_count_bytes,  cudaHostAllocDefault);
}

static void destroyDoubleBufferSlot(DoubleBufferSlot& slot) {
    if (slot.in_flight) cudaEventSynchronize(slot.done);
    slot.e2e_ctx.reset();

    if (slot.d_points) cudaFree(slot.d_points);
    if (slot.h_points_pinned) cudaFreeHost(slot.h_points_pinned);

    if (slot.d_voxel_point_count) cudaFree(slot.d_voxel_point_count);
    if (slot.d_voxels_dense) cudaFree(slot.d_voxels_dense);
    if (slot.d_voxel_features_compact) cudaFree(slot.d_voxel_features_compact);
    if (slot.d_voxel_num_points) cudaFree(slot.d_voxel_num_points);
    if (slot.d_pillar_num) cudaFree(slot.d_pillar_num);
    if (slot.h_pillar_num) cudaFreeHost(slot.h_pillar_num);

    if (slot.d_cand_buffer) cudaFree(slot.d_cand_buffer);
    if (slot.d_cand_count) cudaFree(slot.d_cand_count);
    if (slot.h_cand_buffer_pinned) cudaFreeHost(slot.h_cand_buffer_pinned);
    if (slot.h_cand_count_pinned) cudaFreeHost(slot.h_cand_count_pinned);

    if (slot.frame_start) cudaEventDestroy(slot.frame_start);
    if (slot.frame_stop) cudaEventDestroy(slot.frame_stop);
    if (slot.h2d_start) cudaEventDestroy(slot.h2d_start);
    if (slot.h2d_stop) cudaEventDestroy(slot.h2d_stop);
    if (slot.pillarize_stop) cudaEventDestroy(slot.pillarize_stop);
    if (slot.infer_start) cudaEventDestroy(slot.infer_start);
    if (slot.infer_stop) cudaEventDestroy(slot.infer_stop);
    if (slot.decode_start) cudaEventDestroy(slot.decode_start);
    if (slot.decode_stop) cudaEventDestroy(slot.decode_stop);
    if (slot.done) cudaEventDestroy(slot.done);
    if (slot.stream) cudaStreamDestroy(slot.stream);
}

static void ensureDoubleBufferDevicePoints(DoubleBufferSlot& slot, int N) {
    if (N <= slot.d_points_capacity) return;
    if (slot.d_points) cudaFree(slot.d_points);
    slot.d_points_capacity = N + (N >> 2);
    cudaMalloc(&slot.d_points, (size_t)slot.d_points_capacity * 5 * sizeof(float));
}

static void submitDoubleBufferFrame(DoubleBufferSlot& slot,
                                    const float* pts_ptr,
                                    int N,
                                    int frame_index,
                                    const std::string& key,
                                    bool measured) {
    slot.frame_index = frame_index;
    slot.key = key;
    slot.measured = measured;

    ensureDoubleBufferDevicePoints(slot, N);

    float* tgt_features = static_cast<float*>(
        slot.e2e_ctx->getDeviceBuffer("pillar_features"));
    float* tgt_mask = static_cast<float*>(
        slot.e2e_ctx->getDeviceBuffer("pillar_mask"));
    int* tgt_coords = static_cast<int*>(
        slot.e2e_ctx->getDeviceBuffer("pillar_coords"));

    nvtxRangePushA("submit_frame");

    if (measured) {
        cudaEventRecord(slot.frame_start, slot.stream);
        cudaEventRecord(slot.h2d_start, slot.stream);
    }

    nvtxRangePushA("h2d");
    cudaMemcpyAsync(slot.d_points, pts_ptr, (size_t)N * 5 * sizeof(float),
                    cudaMemcpyHostToDevice, slot.stream);
    if (measured) cudaEventRecord(slot.h2d_stop, slot.stream);
    nvtxRangePop();

    nvtxRangePushA("pillarize");
    launchPillarizeCuda(
        slot.d_points, N,
        X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
        VX, VY, VZ, BEV_H, BEV_W, MAX_PTS, MAX_PILLARS,
        slot.d_voxel_point_count, slot.d_voxels_dense, slot.d_voxel_features_compact,
        slot.d_voxel_num_points, slot.d_pillar_num,
        tgt_features, tgt_mask, tgt_coords,
        slot.h_pillar_num, slot.stream);
    if (measured) cudaEventRecord(slot.pillarize_stop, slot.stream);
    nvtxRangePop();

    // Dynamic shape 仍需 CPU 读 P；同步范围只覆盖 H2D + pillarize。
    nvtxRangePushA("sync_for_P");
    cudaStreamSynchronize(slot.stream);
    int P = static_cast<int>(*slot.h_pillar_num);
    if (P > MAX_PILLARS) P = MAX_PILLARS;
    nvtxRangePop();

    slot.e2e_ctx->setInputShape("pillar_features", {P, MAX_PTS, C_IN});
    slot.e2e_ctx->setInputShape("pillar_mask",     {P, MAX_PTS});
    slot.e2e_ctx->setInputShape("pillar_coords",   {P, 4});

    nvtxRangePushA("infer");
    if (measured) cudaEventRecord(slot.infer_start, slot.stream);
    slot.e2e_ctx->inferAsync();
    if (measured) cudaEventRecord(slot.infer_stop, slot.stream);
    nvtxRangePop();

    nvtxRangePushA("decode");
    if (measured) cudaEventRecord(slot.decode_start, slot.stream);
    cudaMemsetAsync(slot.d_cand_count, 0, sizeof(unsigned int) * 6, slot.stream);
    int cls_offset_gpu = 0;
    for (int t = 0; t < 6; t++) {
        std::string pre = "task" + std::to_string(t) + "_";
        int nc = TASK_NUM_CLS[t];
        auto hm_shape = slot.e2e_ctx->getOutputShape(pre + "hm");
        int H = hm_shape[2], W = hm_shape[3];

        const float* d_hm = static_cast<const float*>(
            slot.e2e_ctx->getDeviceBuffer(pre + "hm"));
        const float* d_center = static_cast<const float*>(
            slot.e2e_ctx->getDeviceBuffer(pre + "center"));
        const float* d_center_z = static_cast<const float*>(
            slot.e2e_ctx->getDeviceBuffer(pre + "center_z"));
        const float* d_dim = static_cast<const float*>(
            slot.e2e_ctx->getDeviceBuffer(pre + "dim"));
        const float* d_rot = static_cast<const float*>(
            slot.e2e_ctx->getDeviceBuffer(pre + "rot"));
        const float* d_vel = static_cast<const float*>(
            slot.e2e_ctx->getDeviceBuffer(pre + "vel"));

        launchDecodeTaskCuda(
            d_hm, d_center, d_center_z, d_dim, d_rot, d_vel,
            H, W, nc, t, cls_offset_gpu, MAX_CANDS_PER_TASK,
            SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE,
            slot.d_cand_buffer + (size_t)t * MAX_CANDS_PER_TASK,
            slot.d_cand_count + t,
            slot.stream);
        cls_offset_gpu += nc;
    }

    cudaMemcpyAsync(slot.h_cand_count_pinned, slot.d_cand_count,
                    sizeof(unsigned int) * 6,
                    cudaMemcpyDeviceToHost, slot.stream);
    cudaMemcpyAsync(slot.h_cand_buffer_pinned, slot.d_cand_buffer,
                    sizeof(Detection) * 6 * MAX_CANDS_PER_TASK,
                    cudaMemcpyDeviceToHost, slot.stream);

    if (measured) {
        cudaEventRecord(slot.decode_stop, slot.stream);
    }
    cudaEventRecord(slot.done, slot.stream);
    nvtxRangePop();   // decode
    nvtxRangePop();   // submit_frame
    slot.in_flight = true;
}

struct DoubleBufferSamples {
    std::vector<float> h2d_points;
    std::vector<float> pillarize;
    std::vector<float> infer_e2e;
    std::vector<float> decode_nms;
    std::vector<float> end_to_end;
    int submitted_frames = 0;
    int collected_frames = 0;
};

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static std::vector<Detection> collectDoubleBufferFrame(DoubleBufferSlot& slot,
                                                       DoubleBufferSamples* samples) {
    cudaEventSynchronize(slot.done);

    const bool record_samples = samples && slot.measured;
    float h2d_ms = 0.0f;
    float pillarize_ms = 0.0f;
    float infer_ms = 0.0f;
    float decode_gpu_ms = 0.0f;
    float e2e_ms = 0.0f;
    if (record_samples) {
        h2d_ms = elapsedMs(slot.h2d_start, slot.h2d_stop);
        pillarize_ms = elapsedMs(slot.h2d_stop, slot.pillarize_stop);
        infer_ms = elapsedMs(slot.infer_start, slot.infer_stop);
        decode_gpu_ms = elapsedMs(slot.decode_start, slot.decode_stop);
    }

    auto nms_start = std::chrono::steady_clock::now();

    std::vector<Detection> all_dets;
    for (int t = 0; t < 6; t++) {
        unsigned int K = slot.h_cand_count_pinned[t];
        if (K > (unsigned int)MAX_CANDS_PER_TASK) {
            std::cerr << "[warn] task " << t << " candidate overflow: "
                      << K << " > " << MAX_CANDS_PER_TASK
                      << "（候选被截断；考虑调大 MAX_CANDS_PER_TASK）\n";
            K = MAX_CANDS_PER_TASK;
        }
        std::vector<Detection> dets(
            slot.h_cand_buffer_pinned + (size_t)t * MAX_CANDS_PER_TASK,
            slot.h_cand_buffer_pinned + (size_t)t * MAX_CANDS_PER_TASK + K);
        dets = circleNMS(dets, TASK_NMS_RADIUS[t]);
        all_dets.insert(all_dets.end(), dets.begin(), dets.end());
    }
    auto nms_stop = std::chrono::steady_clock::now();

    if (record_samples) {
        float nms_ms = std::chrono::duration<float, std::milli>(
            nms_stop - nms_start).count();
        cudaEventRecord(slot.frame_stop, slot.stream);
        cudaEventSynchronize(slot.frame_stop);
        e2e_ms = elapsedMs(slot.frame_start, slot.frame_stop);
        samples->h2d_points.push_back(h2d_ms);
        samples->pillarize.push_back(pillarize_ms);
        samples->infer_e2e.push_back(infer_ms);
        samples->decode_nms.push_back(decode_gpu_ms + nms_ms);
        samples->end_to_end.push_back(e2e_ms);
        samples->collected_frames++;
    }

    slot.in_flight = false;
    slot.measured = false;
    return all_dets;
}

struct DoubleBufferStats {
    float mean = 0.0f;
    float median = 0.0f;
    float p99 = 0.0f;
    float vmin = 0.0f;
    float vmax = 0.0f;
};

static DoubleBufferStats computeDoubleBufferStats(std::vector<float> samples) {
    DoubleBufferStats s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.vmin = samples.front();
    s.vmax = samples.back();
    s.median = samples[samples.size() / 2];
    size_t p99i = static_cast<size_t>(std::min<float>(
        samples.size() - 1, std::floor(samples.size() * 0.99f)));
    s.p99 = samples[p99i];
    s.mean = std::accumulate(samples.begin(), samples.end(), 0.0f) / samples.size();
    return s;
}

static void printDoubleBufferSummary(const DoubleBufferSamples& samples,
                                     float wall_ms) {
    if (samples.end_to_end.empty()) {
        std::cout << "DoubleBuffer benchmark: no samples\n";
        return;
    }
    auto e2e = computeDoubleBufferStats(samples.end_to_end);
    float latency_fps = e2e.median > 0.0f ? 1000.0f / e2e.median : 0.0f;
    float throughput_fps = wall_ms > 0.0f
        ? (samples.collected_frames * 1000.0f / wall_ms)
        : 0.0f;

    std::cout << "\n=== Benchmark (iters=" << samples.collected_frames
              << ", 4b double-buffer, stream cudaEvent + wall clock) ===\n";
    std::cout << std::left << std::setw(18) << "Stage"
              << std::right
              << std::setw(9) << "mean"
              << std::setw(9) << "median"
              << std::setw(9) << "min"
              << std::setw(9) << "max"
              << std::setw(9) << "p99"
              << std::setw(9) << "share"
              << "\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3);

    auto print_row = [&](const std::string& name, const std::vector<float>& values) {
        auto st = computeDoubleBufferStats(values);
        float share = e2e.median > 0.0f ? (st.median / e2e.median * 100.0f) : 0.0f;
        std::cout << std::left << std::setw(18) << name
                  << std::right
                  << std::setw(9) << st.mean
                  << std::setw(9) << st.median
                  << std::setw(9) << st.vmin
                  << std::setw(9) << st.vmax
                  << std::setw(9) << st.p99
                  << std::setw(8) << std::setprecision(1) << share << "%"
                  << std::setprecision(3)
                  << "\n";
    };

    print_row("h2d_points", samples.h2d_points);
    print_row("pillarize", samples.pillarize);
    print_row("infer_e2e", samples.infer_e2e);
    print_row("decode_nms", samples.decode_nms);
    std::cout << std::string(72, '-') << "\n";
    print_row("end_to_end", samples.end_to_end);
    std::cout << std::fixed << std::setprecision(3)
              << "\nsubmitted_frames: " << samples.submitted_frames
              << "  collected_frames: " << samples.collected_frames << "\n"
              << "wall_time_ms: " << wall_ms << "\n"
              << std::setprecision(1)
              << "Throughput FPS (collected / wall_time): " << throughput_fps << " fps\n"
              << "Latency FPS (1000 / median end_to_end): " << latency_fps
              << " fps  (latency median " << std::setprecision(2)
              << e2e.median << " ms)\n";
}

static void writeDoubleBufferJson(const std::string& path,
                                  const DoubleBufferSamples& samples,
                                  float wall_ms) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("无法写入 bench JSON: " + path);

    auto e2e = computeDoubleBufferStats(samples.end_to_end);
    auto h2d = computeDoubleBufferStats(samples.h2d_points);
    auto pillarize = computeDoubleBufferStats(samples.pillarize);
    auto infer = computeDoubleBufferStats(samples.infer_e2e);
    auto decode = computeDoubleBufferStats(samples.decode_nms);
    float latency_fps = e2e.median > 0.0f ? 1000.0f / e2e.median : 0.0f;
    float throughput_fps = wall_ms > 0.0f
        ? (samples.collected_frames * 1000.0f / wall_ms)
        : 0.0f;

    auto write_stats = [&](const DoubleBufferStats& st) {
        f << "{\"mean\": " << st.mean
          << ", \"median\": " << st.median
          << ", \"min\": " << st.vmin
          << ", \"max\": " << st.vmax
          << ", \"p99\": " << st.p99 << "}";
    };

    f << "{\n";
    f << "  \"buffering\": \"double\",\n";
    f << "  \"mode\": \"4b\",\n";
    f << "  \"frames_in_flight\": 2,\n";
    f << "  \"iters\": " << samples.collected_frames << ",\n";
    f << "  \"submitted_frames\": " << samples.submitted_frames << ",\n";
    f << "  \"collected_frames\": " << samples.collected_frames << ",\n";
    f << std::fixed << std::setprecision(4);
    f << "  \"wall_time_ms\": " << wall_ms << ",\n";
    f << "  \"throughput_fps\": " << throughput_fps << ",\n";
    f << "  \"latency_fps_median\": " << latency_fps << ",\n";
    f << "  \"stages\": {\n";
    f << "    \"h2d_points\": "; write_stats(h2d); f << ",\n";
    f << "    \"pillarize\": "; write_stats(pillarize); f << ",\n";
    f << "    \"infer_e2e\": "; write_stats(infer); f << ",\n";
    f << "    \"decode_nms\": "; write_stats(decode); f << ",\n";
    f << "    \"end_to_end\": "; write_stats(e2e); f << "\n";
    f << "  }\n";
    f << "}\n";
}

static int runDoubleBuffer4b(const std::string& e2e_engine_path,
                             const std::vector<std::string>& bin_files,
                             bool benchmark,
                             int warmup_iters,
                             int bench_iters,
                             const std::string& bench_output,
                             const std::string& json_path) {
    std::cerr << "[pipeline] mode=4b double-buffer  pillarize=CUDA postprocess=CUDA pinned-points=on\n";
    centerpoint_trt::registerPillarScatterPlugin();

    // engine 加载一次，两个 slot 各持自己的 IExecutionContext + buffers
    InferEngine shared_engine(e2e_engine_path);

    DoubleBufferSlot slots[2];
    initDoubleBufferSlot(slots[0], shared_engine);
    initDoubleBufferSlot(slots[1], shared_engine);

    auto collectIfNeeded = [&](DoubleBufferSlot& slot,
                               DoubleBufferSamples* samples) -> std::vector<Detection> {
        if (!slot.in_flight) return {};
        return collectDoubleBufferFrame(slot, samples);
    };

    if (benchmark) {
        std::cerr << "[benchmark] preloading " << bin_files.size()
                  << " bin files into pinned memory (excluded from timing)...\n";
        std::vector<float*> preloaded_pinned;
        std::vector<int> preloaded_N;
        preloaded_pinned.reserve(bin_files.size());
        preloaded_N.reserve(bin_files.size());

        size_t total_pinned_bytes = 0;
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
        std::cerr << "[benchmark] warmup=" << warmup_iters
                  << "  iters=" << bench_iters << "\n";

        int n_files = (int)preloaded_N.size();
        for (int i = 0; i < warmup_iters; i++) {
            DoubleBufferSlot& slot = slots[i % 2];
            collectIfNeeded(slot, nullptr);
            int idx = i % n_files;
            submitDoubleBufferFrame(slot, preloaded_pinned[idx], preloaded_N[idx],
                                    i, bin_files[idx], false);
        }
        collectIfNeeded(slots[0], nullptr);
        collectIfNeeded(slots[1], nullptr);

        DoubleBufferSamples samples;
        samples.h2d_points.reserve(bench_iters);
        samples.pillarize.reserve(bench_iters);
        samples.infer_e2e.reserve(bench_iters);
        samples.decode_nms.reserve(bench_iters);
        samples.end_to_end.reserve(bench_iters);
        nvtxRangePushA("bench_window");
        auto bench_start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_iters; i++) {
            DoubleBufferSlot& slot = slots[i % 2];
            collectIfNeeded(slot, &samples);
            int idx = i % n_files;
            submitDoubleBufferFrame(slot, preloaded_pinned[idx], preloaded_N[idx],
                                    i, bin_files[idx], true);
            samples.submitted_frames++;
        }
        collectIfNeeded(slots[0], &samples);
        collectIfNeeded(slots[1], &samples);
        auto bench_stop = std::chrono::steady_clock::now();
        nvtxRangePop();   // bench_window
        float wall_ms = std::chrono::duration<float, std::milli>(
            bench_stop - bench_start).count();

        printDoubleBufferSummary(samples, wall_ms);
        if (!bench_output.empty()) {
            writeDoubleBufferJson(bench_output, samples, wall_ms);
            std::cerr << "[benchmark] wrote " << bench_output << "\n";
        }

        for (auto* p : preloaded_pinned) cudaFreeHost(p);
    } else {
        std::ofstream jf;
        if (!json_path.empty()) {
            jf.open(json_path);
            jf << "{\n";
        }

        auto emit = [&](int frame_index,
                        const std::string& key,
                        const std::vector<Detection>& dets) {
            int total = (int)bin_files.size();
            if (!json_path.empty()) {
                writeFrameJson(jf, key, dets);
                if (frame_index + 1 < total) jf << ",";
                jf << "\n";
            } else {
                std::cout << "\n[" << (frame_index + 1) << "/" << total << "] "
                          << key << "  dets=" << dets.size() << "\n";
                for (const auto& d : dets) {
                    printf("  %-25s score=%.3f  xyz=(%.1f,%.1f,%.1f)  wlh=(%.1f,%.1f,%.1f)  rot=%.2f\n",
                           CLASS_NAMES[d.cls_id], d.score,
                           d.x, d.y, d.z, d.w, d.l, d.h, d.rot);
                }
            }
        };

        int total = (int)bin_files.size();
        for (int fi = 0; fi < total; fi++) {
            DoubleBufferSlot& slot = slots[fi % 2];
            if (slot.in_flight) {
                int done_index = slot.frame_index;
                std::string done_key = slot.key;
                auto dets = collectDoubleBufferFrame(slot, nullptr);
                emit(done_index, done_key, dets);
            }

            int N = loadBinIntoPinned(bin_files[fi],
                                      slot.h_points_pinned,
                                      slot.h_points_cap);
            submitDoubleBufferFrame(slot, slot.h_points_pinned, N,
                                    fi, bin_files[fi], false);
        }

        int start = std::max(0, total - 2);
        for (int fi = start; fi < total; fi++) {
            DoubleBufferSlot& slot = slots[fi % 2];
            if (!slot.in_flight) continue;
            int done_index = slot.frame_index;
            std::string done_key = slot.key;
            auto dets = collectDoubleBufferFrame(slot, nullptr);
            emit(done_index, done_key, dets);
        }

        if (!json_path.empty()) {
            jf << "}\n";
            std::cerr << "JSON written to " << json_path << "\n";
        }
    }

    destroyDoubleBufferSlot(slots[0]);
    destroyDoubleBufferSlot(slots[1]);
    return 0;
}

int main(int argc, char** argv) {
    Mode mode = M4A;
    std::string pfn_engine_path = "/workspace/engines/pfn_fp16.plan";
    std::string bb_engine_path  = "/workspace/engines/backbone_head_fp16.plan";
    std::string e2e_engine_path = "/workspace/engines/centerpoint_e2e_fp16.plan";
    std::string json_path;
    std::string file_list_path;
    std::vector<std::string> bin_files;

    bool benchmark = false;
    int  warmup_iters = 10;
    int  bench_iters = 200;
    std::string bench_output;
    bool cuda_pillarize = false;
    bool cuda_postprocess = false;
    bool pinned_points = false;
    bool double_buffer = false;
    std::string dump_calib_dir;
    int dump_calib_frames = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if      (m == "4a") mode = M4A;
            else if (m == "4b") mode = M4B;
            else { std::cerr << "未知 mode: " << m << "（可选: 4a, 4b）\n"; return 1; }
        }
        else if (arg == "--pfn-engine" && i + 1 < argc) pfn_engine_path = argv[++i];
        else if (arg == "--bb-engine"  && i + 1 < argc) bb_engine_path  = argv[++i];
        else if (arg == "--e2e-engine" && i + 1 < argc) e2e_engine_path = argv[++i];
        else if (arg == "--json"       && i + 1 < argc) json_path = argv[++i];
        else if (arg == "--file-list"  && i + 1 < argc) file_list_path = argv[++i];
        else if (arg == "--benchmark") benchmark = true;
        else if (arg == "--warmup"     && i + 1 < argc) warmup_iters = std::stoi(argv[++i]);
        else if (arg == "--bench-iters" && i + 1 < argc) bench_iters = std::stoi(argv[++i]);
        else if (arg == "--bench-output" && i + 1 < argc) bench_output = argv[++i];
        else if (arg == "--cuda-pillarize")   cuda_pillarize = true;
        else if (arg == "--cuda-postprocess") cuda_postprocess = true;
        else if (arg == "--pinned-points")    pinned_points = true;
        else if (arg == "--double-buffer")    double_buffer = true;
        else if (arg == "--dump-calib-dir" && i + 1 < argc) dump_calib_dir = argv[++i];
        else if (arg == "--calib-frames"   && i + 1 < argc) dump_calib_frames = std::stoi(argv[++i]);
        else bin_files.push_back(arg);
    }

    if (!file_list_path.empty()) {
        std::ifstream lf(file_list_path);
        if (!lf) throw std::runtime_error("无法打开 file list: " + file_list_path);
        std::string line;
        while (std::getline(lf, line)) {
            if (!line.empty()) bin_files.push_back(line);
        }
    }

    if (bin_files.empty()) {
        std::cerr << "Usage: infer_pipeline [--mode 4a|4b]  <file.bin> [...]\n"
                  << "                       [--file-list files.txt]\n"
                  << "                       [--pfn-engine PATH] [--bb-engine PATH]    (4a)\n"
                  << "                       [--e2e-engine PATH]                       (4b)\n"
                  << "                       [--json out.json]\n"
                  << "                       [--cuda-pillarize] [--cuda-postprocess] [--pinned-points] [--double-buffer]\n"
                  << "                       [--benchmark [--warmup N] [--bench-iters N] [--bench-output bench.json]]\n"
                  << "                       [--dump-calib-dir <dir> [--calib-frames N]]\n";
        return 1;
    }
    if (!dump_calib_dir.empty() && dump_calib_frames <= 0) dump_calib_frames = 32;
    int dump_calib_count = 0;

    if (double_buffer) {
        if (mode != M4B) {
            std::cerr << "--double-buffer 当前只支持 --mode 4b 端到端 engine\n";
            return 1;
        }
        if (!cuda_pillarize || !cuda_postprocess || !pinned_points) {
            std::cerr << "--double-buffer requires --cuda-pillarize --cuda-postprocess --pinned-points\n";
            return 1;
        }
        if (!dump_calib_dir.empty()) {
            std::cerr << "--double-buffer 当前不支持 --dump-calib-dir；请用单 buffer 路径 dump 校准输入\n";
            return 1;
        }
        return runDoubleBuffer4b(e2e_engine_path, bin_files, benchmark,
                                 warmup_iters, bench_iters, bench_output, json_path);
    }

    std::cerr << "[pipeline] mode=" << (mode == M4A ? "4a" : "4b") << "\n";

    // ── Engine 加载 ─────────────────────────────────────────────────────────
    std::unique_ptr<InferEngine> pfn_engine, bb_engine, e2e_engine;
    std::unique_ptr<InferContext> pfn_ctx, bb_ctx, e2e_ctx;
    if (mode == M4A) {
        pfn_engine = std::make_unique<InferEngine>(pfn_engine_path);
        bb_engine  = std::make_unique<InferEngine>(bb_engine_path);
        pfn_ctx = std::make_unique<InferContext>(*pfn_engine);
        bb_ctx  = std::make_unique<InferContext>(*bb_engine);
    } else {
        centerpoint_trt::registerPillarScatterPlugin();
        e2e_engine = std::make_unique<InferEngine>(e2e_engine_path);
        e2e_ctx = std::make_unique<InferContext>(*e2e_engine);
    }

    // 4a 独立 d_coords：scatter kernel 从这里读 pillar→cell 的索引
    int* d_coords = nullptr;
    if (mode == M4A) {
        cudaMalloc(&d_coords, MAX_PILLARS * 4 * sizeof(int));
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // CUDA pillarize 路径需要的 workspace + 点云 device buffer + host pinned 拿 P
    float*        d_points = nullptr;
    int           d_points_capacity = 0;
    unsigned int* d_voxel_point_count = nullptr;
    float*        d_voxels_dense = nullptr;
    float*        d_voxel_features_compact = nullptr;
    unsigned int* d_voxel_num_points = nullptr;
    unsigned int* d_pillar_num = nullptr;
    unsigned int* h_pillar_num = nullptr;   // pinned host
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

    // GPU 后处理 workspace：候选 buffer + 计数器 + host pinned mirror
    Detection*    d_cand_buffer         = nullptr;
    unsigned int* d_cand_count          = nullptr;
    Detection*    h_cand_buffer_pinned  = nullptr;
    unsigned int* h_cand_count_pinned   = nullptr;
    if (cuda_postprocess) {
        PostprocessWorkspaceBytes pwb;
        postprocessCudaWorkspaceBytes(6, MAX_CANDS_PER_TASK, pwb);
        cudaMalloc(&d_cand_buffer,                              pwb.cand_buffer_bytes);
        cudaMalloc(&d_cand_count,                               pwb.cand_count_bytes);
        cudaHostAlloc(&h_cand_buffer_pinned, pwb.cand_buffer_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_cand_count_pinned,  pwb.cand_count_bytes,  cudaHostAllocDefault);
        std::cerr << "[pipeline] postprocess=CUDA  workspace="
                  << (pwb.cand_buffer_bytes + pwb.cand_count_bytes) / 1024
                  << " KB\n";
    } else {
        std::cerr << "[pipeline] postprocess=CPU\n";
    }

    // 单帧 pipeline：可选传入 timer 做分阶段计时；返回当帧检测框
    auto runOneFrame = [&](const float* pts_ptr, int N, StageTimer* timer)
                       -> std::vector<Detection> {
        if (timer) timer->beginFrameGpu(stream);

        // pillarize 输出目标：CPU/CUDA 两条路都把 features/mask/coords 写到这三个指针
        // 4a：features/mask 是 pfn engine 的 input device buffer，coords 是独立 d_coords
        // 4b：三个指针都来自 e2e engine 的 input device buffer
        float* tgt_features;
        float* tgt_mask;
        int*   tgt_coords;
        if (mode == M4A) {
            tgt_features = static_cast<float*>(pfn_ctx->getDeviceBuffer("pillar_features"));
            tgt_mask     = static_cast<float*>(pfn_ctx->getDeviceBuffer("pillar_mask"));
            tgt_coords   = d_coords;
        } else {
            tgt_features = static_cast<float*>(e2e_ctx->getDeviceBuffer("pillar_features"));
            tgt_mask     = static_cast<float*>(e2e_ctx->getDeviceBuffer("pillar_mask"));
            tgt_coords   = static_cast<int*>(e2e_ctx->getDeviceBuffer("pillar_coords"));
        }

        int P = 0;
        PillarBatch batch;  // 仅 CPU pillarize 路径填充

        if (cuda_pillarize) {
            // 点云 H2D（按需扩容）
            if (timer) timer->begin("h2d_points");
            if (N > d_points_capacity) {
                if (d_points) cudaFree(d_points);
                d_points_capacity = N + (N >> 2);  // 多分一点避免下次扩容
                cudaMalloc(&d_points, d_points_capacity * 5 * sizeof(float));
            }
            cudaMemcpyAsync(d_points, pts_ptr, N * 5 * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            if (timer) timer->end("h2d_points");

            // pillarize 直接写到 tgt_*，零中间拷贝
            if (timer) timer->begin("pillarize");
            launchPillarizeCuda(
                d_points, N,
                X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                VX, VY, VZ, BEV_H, BEV_W, MAX_PTS, MAX_PILLARS,
                d_voxel_point_count, d_voxels_dense, d_voxel_features_compact,
                d_voxel_num_points, d_pillar_num,
                tgt_features, tgt_mask, tgt_coords,
                h_pillar_num, stream);
            cudaStreamSynchronize(stream);  // 等 h_pillar_num 可读
            P = static_cast<int>(*h_pillar_num);
            if (P > MAX_PILLARS) P = MAX_PILLARS;
            if (timer) timer->end("pillarize");

            // dump features/mask/coords（CUDA 路径）：从 tgt_* D2H，零 pad 到 OPT_P
            if (!dump_calib_dir.empty() && dump_calib_count < dump_calib_frames) {
                int P_dump = std::min(P, OPT_P);
                std::vector<float> h_features((size_t)OPT_P * MAX_PTS * C_IN, 0.0f);
                std::vector<float> h_mask((size_t)OPT_P * MAX_PTS, 0.0f);
                std::vector<int>   h_coords((size_t)OPT_P * 4, 0);
                cudaMemcpy(h_features.data(), tgt_features,
                           (size_t)P_dump * MAX_PTS * C_IN * sizeof(float),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_mask.data(), tgt_mask,
                           (size_t)P_dump * MAX_PTS * sizeof(float),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_coords.data(), tgt_coords,
                           (size_t)P_dump * 4 * sizeof(int),
                           cudaMemcpyDeviceToHost);
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/frame_%03d_pillar_features.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_features.data(), h_features.size() * sizeof(float));
                snprintf(fn, sizeof(fn), "%s/frame_%03d_pillar_mask.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_mask.data(), h_mask.size() * sizeof(float));
                snprintf(fn, sizeof(fn), "%s/frame_%03d_pillar_coords.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_coords.data(), h_coords.size() * sizeof(int));
            }
        } else {
            if (timer) timer->begin("pillarize");
            batch = pillarize(pts_ptr, N,
                X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                VX, VY, VZ, MAX_PTS, MAX_PILLARS);
            P = batch.P;
            if (timer) timer->end("pillarize");

            // dump features/mask/coords（CPU 路径）：直接从 batch host buffer 读，零 pad 到 OPT_P
            if (!dump_calib_dir.empty() && dump_calib_count < dump_calib_frames) {
                int P_dump = std::min(P, OPT_P);
                std::vector<float> h_features((size_t)OPT_P * MAX_PTS * C_IN, 0.0f);
                std::vector<float> h_mask((size_t)OPT_P * MAX_PTS, 0.0f);
                std::vector<int>   h_coords((size_t)OPT_P * 4, 0);
                std::memcpy(h_features.data(), batch.features.data(),
                            (size_t)P_dump * MAX_PTS * C_IN * sizeof(float));
                std::memcpy(h_mask.data(), batch.mask.data(),
                            (size_t)P_dump * MAX_PTS * sizeof(float));
                std::memcpy(h_coords.data(), batch.coords.data(),
                            (size_t)P_dump * 4 * sizeof(int));
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/frame_%03d_pillar_features.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_features.data(), h_features.size() * sizeof(float));
                snprintf(fn, sizeof(fn), "%s/frame_%03d_pillar_mask.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_mask.data(), h_mask.size() * sizeof(float));
                snprintf(fn, sizeof(fn), "%s/frame_%03d_pillar_coords.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_coords.data(), h_coords.size() * sizeof(int));
            }

            // 4a CPU 路径：coords 单独 H2D 到 standalone d_coords（scatter kernel 读这里）
            // 4b CPU 路径：coords H2D 由下面 e2e_ctx->setInput 完成
            if (mode == M4A) {
                if (timer) timer->begin("h2d_coords");
                cudaMemcpy(d_coords, batch.coords.data(), P * 4 * sizeof(int),
                           cudaMemcpyHostToDevice);
                if (timer) timer->end("h2d_coords");
            }
        }

        // ── 推理（mode-specific） ────────────────────────────────────────────
        InferContext* outputs_ctx = nullptr;
        if (mode == M4A) {
            if (timer) timer->begin("pfn");
            pfn_ctx->setInputShape("pillar_features", {P, MAX_PTS, C_IN});
            pfn_ctx->setInputShape("pillar_mask",     {P, MAX_PTS});
            if (!cuda_pillarize) {
                pfn_ctx->setInput("pillar_features",
                                  batch.features.data(), P * MAX_PTS * C_IN * sizeof(float));
                pfn_ctx->setInput("pillar_mask",
                                  batch.mask.data(),     P * MAX_PTS * sizeof(float));
            }
            // CUDA pillarize 路径：device buffer 已被 launchPillarizeCuda 填好，跳过 setInput
            pfn_ctx->infer();
            if (timer) timer->end("pfn");

            if (timer) timer->begin("scatter");
            float* d_canvas = static_cast<float*>(bb_ctx->getDeviceBuffer("spatial_features"));
            cudaMemset(d_canvas, 0, C_OUT * BEV_H * BEV_W * sizeof(float));
            const float* d_emb = static_cast<const float*>(
                pfn_ctx->getDeviceBuffer("pillar_embeddings"));
            launchPillarScatter(d_emb, d_coords, d_canvas, P, C_OUT, BEV_H, BEV_W, stream);
            cudaStreamSynchronize(stream);
            if (timer) timer->end("scatter");

            // 4a-only: dump backbone 输入（散射后 BEV，[1,64,512,512]），用于 backbone+head INT8 校准
            if (!dump_calib_dir.empty() && dump_calib_count < dump_calib_frames) {
                std::vector<float> h_canvas((size_t)C_OUT * BEV_H * BEV_W);
                cudaMemcpy(h_canvas.data(), d_canvas,
                           h_canvas.size() * sizeof(float), cudaMemcpyDeviceToHost);
                char fn[256];
                snprintf(fn, sizeof(fn), "%s/frame_%03d_spatial_features.bin",
                         dump_calib_dir.c_str(), dump_calib_count);
                writeCalibBin(fn, h_canvas.data(), h_canvas.size() * sizeof(float));
            }

            if (timer) timer->begin("backbone");
            bb_ctx->infer();
            if (timer) timer->end("backbone");

            outputs_ctx = bb_ctx.get();
        } else {  // M4B
            e2e_ctx->setInputShape("pillar_features", {P, MAX_PTS, C_IN});
            e2e_ctx->setInputShape("pillar_mask",     {P, MAX_PTS});
            e2e_ctx->setInputShape("pillar_coords",   {P, 4});
            if (!cuda_pillarize) {
                e2e_ctx->setInput("pillar_features",
                                  batch.features.data(), P * MAX_PTS * C_IN * sizeof(float));
                e2e_ctx->setInput("pillar_mask",
                                  batch.mask.data(),     P * MAX_PTS * sizeof(float));
                e2e_ctx->setInput("pillar_coords",
                                  batch.coords.data(),   P * 4 * sizeof(int));
            }
            // CUDA pillarize 路径：三路 device buffer 已被 launchPillarizeCuda 填好

            if (timer) timer->begin("infer_e2e");
            e2e_ctx->infer();
            if (timer) timer->end("infer_e2e");
            outputs_ctx = e2e_ctx.get();
        }

        // 一帧的所有 binding dump 完成后再 +1（4a 含 spatial_features，4b 只有三路）
        if (!dump_calib_dir.empty() && dump_calib_count < dump_calib_frames) {
            dump_calib_count++;
        }

        // ── Postprocess（mode 无关，从 outputs_ctx 读 head 输出） ────────────
        if (timer) timer->begin("decode_nms");
        std::vector<Detection> all_dets;

        if (cuda_postprocess) {
            // GPU 路径：直接读 head engine 的 device output buffer，
            // 避免 6 task × 6 tensor × 16K float 的 D2H（~1.5 MB → 几百 Detection）
            cudaMemsetAsync(d_cand_count, 0,
                            sizeof(unsigned int) * 6, stream);

            int cls_offset_gpu = 0;
            for (int t = 0; t < 6; t++) {
                std::string pre = "task" + std::to_string(t) + "_";
                int nc = TASK_NUM_CLS[t];
                auto hm_shape = outputs_ctx->getOutputShape(pre + "hm");
                int H = hm_shape[2], W = hm_shape[3];

                const float* d_hm = static_cast<const float*>(
                    outputs_ctx->getDeviceBuffer(pre + "hm"));
                const float* d_center = static_cast<const float*>(
                    outputs_ctx->getDeviceBuffer(pre + "center"));
                const float* d_center_z = static_cast<const float*>(
                    outputs_ctx->getDeviceBuffer(pre + "center_z"));
                const float* d_dim = static_cast<const float*>(
                    outputs_ctx->getDeviceBuffer(pre + "dim"));
                const float* d_rot = static_cast<const float*>(
                    outputs_ctx->getDeviceBuffer(pre + "rot"));
                const float* d_vel = static_cast<const float*>(
                    outputs_ctx->getDeviceBuffer(pre + "vel"));

                launchDecodeTaskCuda(
                    d_hm, d_center, d_center_z, d_dim, d_rot, d_vel,
                    H, W, nc, t, cls_offset_gpu, MAX_CANDS_PER_TASK,
                    SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE,
                    d_cand_buffer + (size_t)t * MAX_CANDS_PER_TASK,
                    d_cand_count + t,
                    stream);
                cls_offset_gpu += nc;
            }

            // 一次性 D2H：count + buffer，到 host pinned
            cudaMemcpyAsync(h_cand_count_pinned, d_cand_count,
                            sizeof(unsigned int) * 6,
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_cand_buffer_pinned, d_cand_buffer,
                            sizeof(Detection) * 6 * MAX_CANDS_PER_TASK,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // host 上每 task sort + circle NMS（K 通常 < 几百，CPU 微秒级）
            for (int t = 0; t < 6; t++) {
                unsigned int K = h_cand_count_pinned[t];
                if (K > (unsigned int)MAX_CANDS_PER_TASK) {
                    std::cerr << "[warn] task " << t << " candidate overflow: "
                              << K << " > " << MAX_CANDS_PER_TASK
                              << "（候选被截断；考虑调大 MAX_CANDS_PER_TASK）\n";
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

                auto hm_shape = outputs_ctx->getOutputShape(pre + "hm");
                int H = hm_shape[2], W = hm_shape[3];
                size_t map = (size_t)(H * W) * sizeof(float);

                std::vector<float> hm(nc*H*W), center(2*H*W), center_z(H*W);
                std::vector<float> dim(3*H*W), rot(2*H*W), vel(2*H*W);

                outputs_ctx->getOutput(pre+"hm",       hm.data(),       nc*map);
                outputs_ctx->getOutput(pre+"center",   center.data(),    2*map);
                outputs_ctx->getOutput(pre+"center_z", center_z.data(),  map);
                outputs_ctx->getOutput(pre+"dim",      dim.data(),       3*map);
                outputs_ctx->getOutput(pre+"rot",      rot.data(),       2*map);
                outputs_ctx->getOutput(pre+"vel",      vel.data(),       2*map);

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
        float*  h_pinned_scratch = nullptr;
        size_t  pinned_scratch_cap = 0;

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

    if (!dump_calib_dir.empty()) {
        std::ofstream mf(dump_calib_dir + "/manifest.txt");
        mf << "num_frames " << dump_calib_count << "\n"
           << "opt_p "      << OPT_P << "\n"
           << "max_pts "    << MAX_PTS << "\n"
           << "c_in "       << C_IN << "\n"
           << "c_out "      << C_OUT << "\n"
           << "bev_h "      << BEV_H << "\n"
           << "bev_w "      << BEV_W << "\n"
           << "pfn_features_shape " << OPT_P << " " << MAX_PTS << " " << C_IN << "\n"
           << "pfn_mask_shape "     << OPT_P << " " << MAX_PTS << "\n"
           << "e2e_coords_shape "   << OPT_P << " 4\n"
           << "bb_input_shape 1 "   << C_OUT << " " << BEV_H << " " << BEV_W << "\n";
        std::cerr << "[calib] dumped " << dump_calib_count
                  << " frames to " << dump_calib_dir << "\n";
    }

    if (mode == M4A && d_coords) cudaFree(d_coords);
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
