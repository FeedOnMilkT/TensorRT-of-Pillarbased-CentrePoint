#include "scatter_kernel.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cassert>

int main() {
    const int P = 3, C = 4, H = 8, W = 8, B = 1;

    // pillar 特征：pillar p 的所有通道值都填 (p+1)*1.0f，方便识别
    std::vector<float> h_feats(P * C);
    for (int p = 0; p < P; p++)
        for (int c = 0; c < C; c++)
            h_feats[p * C + c] = (float)(p + 1);

    // pillar 坐标：(batch, z, y, x)
    std::vector<int> h_coords = {
        0, 0, 2, 3,   // pillar 0 → canvas[:,2,3]
        0, 0, 5, 1,   // pillar 1 → canvas[:,5,1]
        0, 0, 7, 7,   // pillar 2 → canvas[:,7,7]
    };

    // 分配 GPU 内存
    float* d_feats;   cudaMalloc(&d_feats,  P * C * sizeof(float));
    int*   d_coords;  cudaMalloc(&d_coords, P * 4 * sizeof(int));
    float* d_canvas;  cudaMalloc(&d_canvas, B * C * H * W * sizeof(float));

    cudaMemcpy(d_feats,  h_feats.data(),  P * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords, h_coords.data(), P * 4 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemset(d_canvas, 0, B * C * H * W * sizeof(float));

    launchPillarScatter(d_feats, d_coords, d_canvas, P, C, H, W, nullptr);
    cudaDeviceSynchronize();

    // 拷回验证
    std::vector<float> h_canvas(B * C * H * W, 0.f);
    cudaMemcpy(h_canvas.data(), d_canvas, h_canvas.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // canvas[batch, c, y, x] = h_canvas[batch*C*H*W + c*H*W + y*W + x]
    auto at = [&](int c, int y, int x) {
        return h_canvas[c * H * W + y * W + x];
    };

    bool ok = true;
    // pillar 0：值=1.0，位置(y=2,x=3)
    for (int c = 0; c < C; c++) ok &= (at(c,2,3) == 1.0f);
    // pillar 1：值=2.0，位置(y=5,x=1)
    for (int c = 0; c < C; c++) ok &= (at(c,5,1) == 2.0f);
    // pillar 2：值=3.0，位置(y=7,x=7)
    for (int c = 0; c < C; c++) ok &= (at(c,7,7) == 3.0f);
    // 未写入位置应为 0
    ok &= (at(0,0,0) == 0.0f);

    printf("scatter kernel: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_feats);
    cudaFree(d_coords);
    cudaFree(d_canvas);
    return ok ? 0 : 1;
}
