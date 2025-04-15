#include "scatter_kernel.h"
#include <cstdio>

// ── kernel：每个线程处理一个 pillar ──────────────────────────────────────────
//
// CUDA 执行模型：
//   线程（thread）是最小执行单元
//   线程块（block）= 若干线程，共享 shared memory，这里每块 256 个线程
//   网格（grid）   = 若干块，覆盖全部 P 个 pillar
//
//   线程全局编号：p = blockIdx.x * blockDim.x + threadIdx.x
//   只要 p < P，就处理第 p 个 pillar

__global__ void pillarScatterKernel(
    const float* __restrict__ d_pillar_features,  // [P, C]
    const int*   __restrict__ d_pillar_coords,    // [P, 4]: (batch, z, y, x)
    float*                    d_canvas,           // [B, C, H, W]
    int P, int C, int H, int W)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;  // 边界保护：最后一块可能有多余的线程

    // 从坐标数组读出该 pillar 在 BEV 网格中的位置
    int batch = d_pillar_coords[p * 4 + 0];
    int y     = d_pillar_coords[p * 4 + 2];
    int x     = d_pillar_coords[p * 4 + 3];

    // canvas 内存布局是 [B, C, H, W]（NCHW），计算该 pillar 的起始偏移
    // canvas[batch, c, y, x] = batch*C*H*W + c*H*W + y*W + x
    int canvas_base = batch * C * H * W + y * W + x;

    for (int c = 0; c < C; c++)
        d_canvas[canvas_base + c * H * W] = d_pillar_features[p * C + c];
}

// ── host 包装函数：配置 grid/block，启动 kernel ───────────────────────────────

void launchPillarScatter(
    const float* d_pillar_features,
    const int*   d_pillar_coords,
    float*       d_canvas,
    int P, int C, int H, int W,
    cudaStream_t stream)
{
    // 每块 256 个线程，向上取整保证覆盖全部 P 个 pillar
    const int threads = 256;
    const int blocks  = (P + threads - 1) / threads;

    pillarScatterKernel<<<blocks, threads, 0, stream>>>(
        d_pillar_features, d_pillar_coords, d_canvas, P, C, H, W);
}
