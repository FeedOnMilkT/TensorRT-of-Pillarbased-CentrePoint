#pragma once
#include <cuda_runtime.h>

struct PillarizeWorkspaceBytes {
    size_t voxel_point_count_bytes;   // 阶段 1 的取号计数器 [H*W]，跟输出 pillar_mask 区分
    size_t voxels_dense_bytes;
    size_t voxel_features_bytes;
    size_t voxel_num_points_bytes;
    size_t pillar_num_bytes;
};

void pillarizeCudaWorkspaceBytes(int H, int W, int max_pts, int max_pillars,
                                 PillarizeWorkspaceBytes& out);

// CUDA pillarization。所有 device buffer 由调用方分配；workspace buffer 内部会
// 在每次调用时按需 memset。h_pillar_num 在 stream 同步后才有效。
//
// 输出布局与 CPU 版 pillarize() 完全一致：
//   d_features_out     [max_pillars, max_pts, 11]
//   d_pillar_mask_out  [max_pillars, max_pts]            1=有效点 0=填充
//   d_coords_out       [max_pillars, 4]                  (b=0, z=0, y, x)
//
// 注意：pillar 的 id 分配顺序由 atomicAdd 决定，与 CPU 版（hash map 遍历顺序）不同；
// 同一 pillar 内点的顺序也可能不同（atomicAdd 抢号）。但每个 (y, x) 对应的有效点
// 集合（multiset）一致，PFN 的 pooled embedding 不受影响。
void launchPillarizeCuda(
        const float* d_points, int N,
        float x_min, float x_max,
        float y_min, float y_max,
        float z_min, float z_max,
        float vx, float vy, float vz,
        int H, int W,
        int max_pts, int max_pillars,
        unsigned int* d_voxel_point_count_workspace,
        float* d_voxels_dense_workspace,
        float* d_voxel_features_workspace,
        unsigned int* d_voxel_num_points_workspace,
        unsigned int* d_pillar_num_workspace,
        float* d_features_out,
        float* d_pillar_mask_out,
        int* d_coords_out,
        unsigned int* h_pillar_num,
        cudaStream_t stream);
