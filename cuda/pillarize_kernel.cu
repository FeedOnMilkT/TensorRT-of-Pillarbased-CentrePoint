#include "pillarize_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>

// 三段式 GPU pillarization，参考 NVIDIA TensorRT plugin 的 voxelGeneratorKernels.cu。
//
// 阶段 1 voxelizePointsKernel：每个点一个线程，按 (px, py) 算 voxel_id；
//        atomicAdd(voxel_point_count[voxel_id]) 拿到该点在 voxel 内的局部序号（取号机模式）；
//        若 < max_pts，atomicExch 写到 dense voxel grid。
// 阶段 2 compactPillarsKernel：每个 voxel 一个线程，非空 voxel 用 atomicAdd(pillar_num)
//        分配 pillar id，紧凑成 [P, max_pts, 5] + coords[P,4] + voxel_num_points[P]。
// 阶段 3 computeFeaturesKernel：每个 pillar 由 max_pts 个线程协作处理，
//        shared memory 算 pillar 内点云重心，输出 11 维 features + pillar_mask。
//
// max_pts 当前固定 20，max_pts 必须 <= 32 以便保证一个 pillar 一个 warp 内同步。
//
// 命名约定：
//   - voxel_point_count [H*W] —— 阶段 1 的取号计数器（每个 voxel 已被多少个点抢号）
//   - pillar_mask [P, max_pts] —— 阶段 3 输出给 PFN 的有效位掩码（1.0=真实点 0.0=填充）
// 这两个不同性质的字段在 NVIDIA 原版都叫 mask，本实现拆开命名避免混淆。

namespace {

constexpr int NUM_INPUT_FEATS = 5;   // x, y, z, intensity, time
constexpr int NUM_OUTPUT_FEATS = 11;
constexpr int PILLARS_PER_BLOCK = 4;
constexpr int MAX_PTS_LIMIT = 32;

__global__ void voxelizePointsKernel(
        const float* __restrict__ points, int N,
        float x_min, float x_max,
        float y_min, float y_max,
        float z_min, float z_max,
        float vx, float vy,
        int H, int W,
        int max_pts,
        unsigned int* __restrict__ voxel_point_count,  // [H*W] 取号计数器
        float* __restrict__ voxels)                    // [H*W, max_pts, 5]
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float px = points[n * NUM_INPUT_FEATS + 0];
    float py = points[n * NUM_INPUT_FEATS + 1];
    float pz = points[n * NUM_INPUT_FEATS + 2];
    float pi = points[n * NUM_INPUT_FEATS + 3];
    float pt = points[n * NUM_INPUT_FEATS + 4];

    if (px < x_min || px >= x_max) return;
    if (py < y_min || py >= y_max) return;
    if (pz < z_min || pz >= z_max) return;

    int vox_x = static_cast<int>(floorf((px - x_min) / vx));
    int vox_y = static_cast<int>(floorf((py - y_min) / vy));
    if (vox_x >= W) vox_x = W - 1;
    if (vox_y >= H) vox_y = H - 1;

    int voxel_id = vox_y * W + vox_x;
    // 取号：原子地拿到该点在该 voxel 内的局部序号；号 >= max_pts 直接丢弃
    unsigned int point_id = atomicAdd(&voxel_point_count[voxel_id], 1u);
    if (point_id >= static_cast<unsigned int>(max_pts)) return;

    float* dst = voxels + (voxel_id * max_pts + point_id) * NUM_INPUT_FEATS;
    atomicExch(dst + 0, px);
    atomicExch(dst + 1, py);
    atomicExch(dst + 2, pz);
    atomicExch(dst + 3, pi);
    atomicExch(dst + 4, pt);
}

__global__ void compactPillarsKernel(
        const unsigned int* __restrict__ voxel_point_count,  // [H*W] 阶段 1 留下的取号终值
        const float* __restrict__ voxels,                    // [H*W, max_pts, 5]
        int H, int W,
        int max_pts,
        int max_pillars,
        unsigned int* __restrict__ pillar_num,        // [1]
        float* __restrict__ voxel_features_compact,   // [max_pillars, max_pts, 5]
        unsigned int* __restrict__ voxel_num_points,  // [max_pillars]
        int* __restrict__ coords)                     // [max_pillars, 4]: (b=0, z=0, y, x)
{
    int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (voxel_id >= total) return;

    unsigned int count = voxel_point_count[voxel_id];
    if (count == 0) return;
    if (count > static_cast<unsigned int>(max_pts)) count = max_pts;

    unsigned int pid = atomicAdd(pillar_num, 1u);
    if (pid >= static_cast<unsigned int>(max_pillars)) return;   // 截断

    int vy = voxel_id / W;
    int vx = voxel_id % W;
    coords[pid * 4 + 0] = 0;
    coords[pid * 4 + 1] = 0;
    coords[pid * 4 + 2] = vy;
    coords[pid * 4 + 3] = vx;
    voxel_num_points[pid] = count;

    const float* src = voxels + voxel_id * max_pts * NUM_INPUT_FEATS;
    float* dst = voxel_features_compact + pid * max_pts * NUM_INPUT_FEATS;
    for (unsigned int i = 0; i < count; i++) {
        for (int k = 0; k < NUM_INPUT_FEATS; k++) {
            dst[i * NUM_INPUT_FEATS + k] = src[i * NUM_INPUT_FEATS + k];
        }
    }
}

__global__ void computeFeaturesKernel(
        const float* __restrict__ voxel_features_compact,  // [P, max_pts, 5]
        const unsigned int* __restrict__ voxel_num_points, // [P]
        const int* __restrict__ coords,                    // [P, 4]
        const unsigned int* __restrict__ pillar_num,       // [1] 实际 P (未截断的话)
        float x_min, float y_min, float z_min,
        float vx, float vy, float vz,
        int max_pts,
        int max_pillars,
        float* __restrict__ features,        // [max_pillars, max_pts, 11]
        float* __restrict__ pillar_mask)     // [max_pillars, max_pts]  1=valid 0=pad
{
    int warp_size = max_pts;
    int pillar_in_block = threadIdx.x / warp_size;
    int point_idx = threadIdx.x % warp_size;
    int pid = blockIdx.x * PILLARS_PER_BLOCK + pillar_in_block;

    // 越界判定不能 early return：同 block 其他线程仍要走 __syncthreads()
    unsigned int actual_P = pillar_num[0];
    if (actual_P > static_cast<unsigned int>(max_pillars)) actual_P = max_pillars;
    bool pillar_in_range = (pid < max_pillars) && (pid < static_cast<int>(actual_P));
    bool thread_in_range = pillar_in_range && (point_idx < max_pts);

    __shared__ float pillarSM[PILLARS_PER_BLOCK][MAX_PTS_LIMIT][NUM_INPUT_FEATS];
    __shared__ float sumXYZ[PILLARS_PER_BLOCK][3];
    __shared__ int   coordsSM[PILLARS_PER_BLOCK][4];
    __shared__ int   numPtsSM[PILLARS_PER_BLOCK];

    if (point_idx == 0) {
        if (pillar_in_range) {
            numPtsSM[pillar_in_block]   = static_cast<int>(voxel_num_points[pid]);
            coordsSM[pillar_in_block][0] = coords[pid * 4 + 0];
            coordsSM[pillar_in_block][1] = coords[pid * 4 + 1];
            coordsSM[pillar_in_block][2] = coords[pid * 4 + 2];
            coordsSM[pillar_in_block][3] = coords[pid * 4 + 3];
        } else {
            numPtsSM[pillar_in_block]   = 0;
            coordsSM[pillar_in_block][2] = 0;
            coordsSM[pillar_in_block][3] = 0;
        }
        sumXYZ[pillar_in_block][0] = 0.f;
        sumXYZ[pillar_in_block][1] = 0.f;
        sumXYZ[pillar_in_block][2] = 0.f;
    }
    if (thread_in_range) {
        for (int k = 0; k < NUM_INPUT_FEATS; k++) {
            pillarSM[pillar_in_block][point_idx][k] =
                voxel_features_compact[(pid * max_pts + point_idx) * NUM_INPUT_FEATS + k];
        }
    }
    __syncthreads();

    int npts = numPtsSM[pillar_in_block];
    bool is_valid = thread_in_range && (point_idx < npts);

    if (is_valid) {
        atomicAdd(&sumXYZ[pillar_in_block][0], pillarSM[pillar_in_block][point_idx][0]);
        atomicAdd(&sumXYZ[pillar_in_block][1], pillarSM[pillar_in_block][point_idx][1]);
        atomicAdd(&sumXYZ[pillar_in_block][2], pillarSM[pillar_in_block][point_idx][2]);
    }
    __syncthreads();

    if (!thread_in_range) return;

    float fnpts = (npts > 0) ? static_cast<float>(npts) : 1.f;
    float mx = sumXYZ[pillar_in_block][0] / fnpts;
    float my = sumXYZ[pillar_in_block][1] / fnpts;
    float mz = sumXYZ[pillar_in_block][2] / fnpts;

    int cy = coordsSM[pillar_in_block][2];
    int cx = coordsSM[pillar_in_block][3];
    float xc = vx * 0.5f + cx * vx + x_min;
    float yc = vy * 0.5f + cy * vy + y_min;
    float zc = vz * 0.5f + 0  * vz + z_min;

    float* fout = features + (pid * max_pts + point_idx) * NUM_OUTPUT_FEATS;
    if (is_valid) {
        float x = pillarSM[pillar_in_block][point_idx][0];
        float y = pillarSM[pillar_in_block][point_idx][1];
        float z = pillarSM[pillar_in_block][point_idx][2];
        fout[0]  = x;
        fout[1]  = y;
        fout[2]  = z;
        fout[3]  = pillarSM[pillar_in_block][point_idx][3];
        fout[4]  = pillarSM[pillar_in_block][point_idx][4];
        fout[5]  = x - mx;
        fout[6]  = y - my;
        fout[7]  = z - mz;
        fout[8]  = x - xc;
        fout[9]  = y - yc;
        fout[10] = z - zc;
        pillar_mask[pid * max_pts + point_idx] = 1.f;
    } else {
        for (int k = 0; k < NUM_OUTPUT_FEATS; k++) fout[k] = 0.f;
        pillar_mask[pid * max_pts + point_idx] = 0.f;
    }
}

}  // namespace

void pillarizeCudaWorkspaceBytes(int H, int W, int max_pts, int max_pillars,
                                 PillarizeWorkspaceBytes& out)
{
    out.voxel_point_count_bytes  = sizeof(unsigned int) * H * W;
    out.voxels_dense_bytes       = sizeof(float) * H * W * max_pts * NUM_INPUT_FEATS;
    out.voxel_features_bytes     = sizeof(float) * max_pillars * max_pts * NUM_INPUT_FEATS;
    out.voxel_num_points_bytes   = sizeof(unsigned int) * max_pillars;
    out.pillar_num_bytes         = sizeof(unsigned int);
}

void launchPillarizeCuda(
        const float* d_points, int N,
        float x_min, float x_max,
        float y_min, float y_max,
        float z_min, float z_max,
        float vx, float vy, float vz,
        int H, int W,
        int max_pts, int max_pillars,
        // workspace（调用方已分配）
        unsigned int* d_voxel_point_count_workspace,
        float* d_voxels_dense_workspace,
        float* d_voxel_features_workspace,
        unsigned int* d_voxel_num_points_workspace,
        unsigned int* d_pillar_num_workspace,
        // 输出
        float* d_features_out,        // [max_pillars, max_pts, 11]
        float* d_pillar_mask_out,     // [max_pillars, max_pts]
        int* d_coords_out,            // [max_pillars, 4]
        unsigned int* h_pillar_num,   // host int, 输出实际 P
        cudaStream_t stream)
{
    // 0. 清零取号计数器和 pillar_num（voxels_dense / voxel_features 的脏数据
    //    会被 atomicAdd / atomicExch 自然覆盖到合法位置，不需要 memset）
    cudaMemsetAsync(d_voxel_point_count_workspace, 0, sizeof(unsigned int) * H * W, stream);
    cudaMemsetAsync(d_pillar_num_workspace, 0, sizeof(unsigned int), stream);

    // 1. voxelize
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        voxelizePointsKernel<<<blocks, threads, 0, stream>>>(
            d_points, N,
            x_min, x_max, y_min, y_max, z_min, z_max,
            vx, vy,
            H, W, max_pts,
            d_voxel_point_count_workspace,
            d_voxels_dense_workspace);
    }

    // 2. compact
    {
        int threads = 1024;
        int total = H * W;
        int blocks = (total + threads - 1) / threads;
        compactPillarsKernel<<<blocks, threads, 0, stream>>>(
            d_voxel_point_count_workspace,
            d_voxels_dense_workspace,
            H, W, max_pts, max_pillars,
            d_pillar_num_workspace,
            d_voxel_features_workspace,
            d_voxel_num_points_workspace,
            d_coords_out);
    }

    // 3. compute features
    {
        int threads = PILLARS_PER_BLOCK * max_pts;
        int blocks = (max_pillars + PILLARS_PER_BLOCK - 1) / PILLARS_PER_BLOCK;
        computeFeaturesKernel<<<blocks, threads, 0, stream>>>(
            d_voxel_features_workspace,
            d_voxel_num_points_workspace,
            d_coords_out,
            d_pillar_num_workspace,
            x_min, y_min, z_min,
            vx, vy, vz,
            max_pts, max_pillars,
            d_features_out,
            d_pillar_mask_out);
    }

    // 4. 把 P 拉回 host（同流上的 D2H，等待 stream 即可）
    cudaMemcpyAsync(h_pillar_num, d_pillar_num_workspace,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
}
