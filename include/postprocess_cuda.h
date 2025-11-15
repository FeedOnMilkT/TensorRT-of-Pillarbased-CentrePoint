#pragma once
#include <cuda_runtime.h>
#include "postprocess.h"  // Detection 结构体

// CUDA-side decode + 3×3 peak filter (单 task)。
// circle NMS 暂留在 CPU 上：peak filter 之后候选数从 H*W*num_cls (~16K-32K) 跌到几百，
// CPU sort + O(K²) NMS 在微秒量级，搬上 GPU 收益边际，先把瓶颈最大的 decode 段 CUDA 化。
//
// 调用约定：
//   1. postprocessCudaWorkspaceBytes() 查 workspace 字节数（含 num_tasks slot）
//   2. 调用方一次性 cudaMalloc 两块 buffer：cand_buffer 和 cand_count
//   3. 每帧开头 cudaMemsetAsync(d_cand_count, 0, count_bytes, stream)
//   4. 对每个 task 调 launchDecodeTaskCuda（同 stream 串行入队）
//   5. 一次性把 cand_count + cand_buffer 拷回 host pinned，stream 同步
//   6. 每个 task 在 host 上 sort + circleNMS 自己 slot 的前 cand_count[task_id] 个候选
//
// 与 CPU decodeTask 的等价性：
//   - peak check 用 logit 直接比较（sigmoid 单调，省 8 次 sigmoid 调用）
//   - score = sigmoid(center_logit) 与 CPU 同步
//   - decode 公式与 CPU 完全一致
//   - 候选输出顺序不同（atomicAdd 抢号），需在 host 上排序后再比较

struct PostprocessWorkspaceBytes {
    size_t cand_buffer_bytes;   // [num_tasks * max_cands_per_task] sizeof(Detection)
    size_t cand_count_bytes;    // [num_tasks] sizeof(unsigned int)
};

void postprocessCudaWorkspaceBytes(int num_tasks, int max_cands_per_task,
                                   PostprocessWorkspaceBytes& out);

// 解码 + peak 过滤一个 task 的 CenterHead 输出。
//   d_hm/center/center_z/dim/rot/vel：[C, H, W] device buffer，可直接来自 TRT engine output
//   d_cand_buffer_task：调用方需偏移到 task slot，例如 base + task_id * max_cands_per_task
//   d_cand_count_task：调用方需偏移到 base + task_id
// 当候选数 > max_cands_per_task 时溢出候选被静默丢弃，但计数器 d_cand_count 仍累加，
// 调用方读到大于 max 的值即为溢出告警。
void launchDecodeTaskCuda(
        const float* d_hm,
        const float* d_center,
        const float* d_center_z,
        const float* d_dim,
        const float* d_rot,
        const float* d_vel,
        int H, int W, int num_cls,
        int task_id, int cls_offset, int max_cands_per_task,
        float score_thresh,
        float x_min, float y_min, float vx, float vy, int out_stride,
        Detection* d_cand_buffer_task,
        unsigned int* d_cand_count_task,
        cudaStream_t stream);
