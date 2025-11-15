#include "postprocess_cuda.h"
#include "postprocess.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// CenterPoint 后处理 GPU 实现：单 task 解码 + 3×3 局部极大值过滤。
// CPU 路径见 src/postprocess.cpp::decodeTask；本 kernel 与之等价（采样顺序差异除外）。
//
// 关键优化：peak check 用 raw logit 比较，sigmoid 单调 → 跟比 sigmoid 等价但省 8 次 sigmoid。
// 这一行 trick 把内层 9 次 expf 压成 1 次（仅最终分数计算）。

namespace {

__device__ __forceinline__ float sigmoidDev(float x) {
    return 1.f / (1.f + __expf(-x));
}

__global__ void decodeTaskKernel(
        const float* __restrict__ hm,         // [num_cls, H, W]
        const float* __restrict__ center,     // [2, H, W]
        const float* __restrict__ center_z,   // [1, H, W]
        const float* __restrict__ dim,        // [3, H, W]
        const float* __restrict__ rot,        // [2, H, W]
        const float* __restrict__ vel,        // [2, H, W]
        int H, int W, int num_cls,
        int task_id, int cls_offset, int max_cands,
        float score_thresh,
        float x_min, float y_min, float vx, float vy, int out_stride,
        Detection* __restrict__ cand_buffer,        // [max_cands]
        unsigned int* __restrict__ cand_count)      // 单计数器
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cls * H * W;
    if (idx >= total) return;

    int hw = H * W;
    int c  = idx / hw;
    int rem = idx - c * hw;
    int gy = rem / W;
    int gx = rem - gy * W;

    int center_off = c * hw + gy * W + gx;
    float center_logit = hm[center_off];

    // 用 sigmoid 阈值，跟 CPU 完全一致（避免 logit 阈值边界数值差异）
    float score = sigmoidDev(center_logit);
    if (score < score_thresh) return;

    // 3×3 peak check：sigmoid 单调，logit 比较与 sigmoid 比较等价
    // 边界外邻居跳过（与 CPU 行为一致：CPU 中 dy/dx 越界 continue）
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        int ny = gy + dy;
        if (ny < 0 || ny >= H) continue;
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0) continue;
            int nx = gx + dx;
            if (nx < 0 || nx >= W) continue;
            float neighbor_logit = hm[c * hw + ny * W + nx];
            if (neighbor_logit > center_logit) return;  // 不是局部最大
        }
    }

    // 通过 → 分配候选 slot
    unsigned int slot = atomicAdd(cand_count, 1u);
    if (slot >= (unsigned int)max_cands) return;  // 溢出但计数器仍累加，host 侧可检测

    int gxy_off = gy * W + gx;

    Detection d;
    d.score   = score;
    d.task_id = task_id;
    d.cls_id  = cls_offset + c;

    // decode 公式与 CPU 完全对齐
    d.x = (gx + center[0 * hw + gxy_off]) * out_stride * vx + x_min;
    d.y = (gy + center[1 * hw + gxy_off]) * out_stride * vy + y_min;
    d.z = center_z[gxy_off];
    d.w = __expf(dim[0 * hw + gxy_off]);
    d.l = __expf(dim[1 * hw + gxy_off]);
    d.h = __expf(dim[2 * hw + gxy_off]);
    d.rot = atan2f(rot[1 * hw + gxy_off], rot[0 * hw + gxy_off]);
    d.vx  = vel[0 * hw + gxy_off];
    d.vy  = vel[1 * hw + gxy_off];

    cand_buffer[slot] = d;
}

}  // namespace

void postprocessCudaWorkspaceBytes(int num_tasks, int max_cands_per_task,
                                   PostprocessWorkspaceBytes& out)
{
    out.cand_buffer_bytes = sizeof(Detection) * (size_t)num_tasks * (size_t)max_cands_per_task;
    out.cand_count_bytes  = sizeof(unsigned int) * (size_t)num_tasks;
}

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
        cudaStream_t stream)
{
    int total = num_cls * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    decodeTaskKernel<<<blocks, threads, 0, stream>>>(
        d_hm, d_center, d_center_z, d_dim, d_rot, d_vel,
        H, W, num_cls, task_id, cls_offset, max_cands_per_task,
        score_thresh,
        x_min, y_min, vx, vy, out_stride,
        d_cand_buffer_task, d_cand_count_task);
}
