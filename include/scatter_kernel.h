#pragma once
#include <cuda_runtime.h>

// 将 PFN 输出的 pillar 特征按坐标写入 BEV 画布
// d_pillar_features : GPU, [P, C]         PFN 输出的 pillar 嵌入
// d_pillar_coords   : GPU, [P, 4]         每个 pillar 的坐标 (batch, z, y, x)
// d_canvas          : GPU, [B, C, H, W]   输出 BEV 伪图像，调用前需清零
// P                 : 本帧实际 pillar 数量
void launchPillarScatter(
    const float* d_pillar_features,
    const int*   d_pillar_coords,
    float*       d_canvas,
    int P, int C, int H, int W,
    cudaStream_t stream);
