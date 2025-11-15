#include "postprocess.h"
#include "postprocess_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// CPU vs CUDA decode + peak filter 等价性单测。
// 用合成数据：随机 logit 分布（mean=-2, sigma=1.5）模拟 CenterHead 输出，
// 大多 cell 会被 score_thresh=0.1 过滤掉，少数通过的再过 3×3 peak 检验。
//
// 比较策略（与 test_pillarize 同模式）：
//   两路输出顺序不同（atomicAdd 抢号），按 (cls_id, gy 按 y 反推, gx 按 x 反推) 不可靠，
//   改成按 (cls_id, score) 排序，逐字段比较。score 在两边数值上对位，所以排序后能稳定配对。

namespace {

constexpr int H = 128, W = 128;
constexpr int NUM_TASKS = 6;
constexpr int TASK_NUM_CLS[6] = {1, 2, 2, 1, 2, 2};
constexpr int CLS_OFFSETS[6] = {0, 1, 3, 5, 6, 8};
constexpr float SCORE_THRESH = 0.1f;
constexpr float X_MIN = -51.2f, Y_MIN = -51.2f;
constexpr float VX = 0.2f, VY = 0.2f;
constexpr int OUT_STRIDE = 4;
constexpr int MAX_CANDS_PER_TASK = 4096;
constexpr float TOL = 1e-4f;

// 把 Detection 反推回 cell 坐标。center sigma 小（< 0.5）时 gx/gy 反推恒精确。
// 注意 d.x = (gx + center_offset) * out_stride * vx + x_min，center_offset ∈ ~[-0.3, 0.3]
// → round((d.x - x_min) / (out_stride * vx)) 总等于 gx。
struct CellKey {
    int cls_id;
    int gx;
    int gy;
};

CellKey toCellKey(const Detection& d) {
    float gx_f = (d.x - X_MIN) / (OUT_STRIDE * VX);
    float gy_f = (d.y - Y_MIN) / (OUT_STRIDE * VY);
    return {d.cls_id, (int)std::lroundf(gx_f), (int)std::lroundf(gy_f)};
}

struct DetKey {
    static bool less(const Detection& a, const Detection& b) {
        auto ka = toCellKey(a), kb = toCellKey(b);
        if (ka.cls_id != kb.cls_id) return ka.cls_id < kb.cls_id;
        if (ka.gy != kb.gy) return ka.gy < kb.gy;
        return ka.gx < kb.gx;
    }
};

}  // namespace

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> hm_dist(-2.0f, 1.5f);   // logit 分布
    std::normal_distribution<float> reg_dist(0.f, 1.f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    PostprocessWorkspaceBytes wb;
    postprocessCudaWorkspaceBytes(NUM_TASKS, MAX_CANDS_PER_TASK, wb);
    Detection* d_cand_buffer = nullptr;
    unsigned int* d_cand_count = nullptr;
    cudaMalloc(&d_cand_buffer, wb.cand_buffer_bytes);
    cudaMalloc(&d_cand_count, wb.cand_count_bytes);
    cudaMemsetAsync(d_cand_count, 0, wb.cand_count_bytes, stream);

    int total_cpu = 0, total_cuda = 0;
    int fail_count = 0;
    float global_max_err = 0.f;

    for (int t = 0; t < NUM_TASKS; t++) {
        int nc = TASK_NUM_CLS[t];
        int hw = H * W;

        std::vector<float> hm(nc * hw), center(2 * hw), center_z(hw),
                           dim(3 * hw), rot(2 * hw), vel(2 * hw);
        for (auto& v : hm)       v = hm_dist(rng);
        // center 限制在 sigma=0.1，保证 |offset| < 0.5 → gx/gy 反推精确
        for (auto& v : center)   v = reg_dist(rng) * 0.1f;
        for (auto& v : center_z) v = reg_dist(rng);
        for (auto& v : dim)      v = reg_dist(rng) * 0.5f;
        for (auto& v : rot)      v = reg_dist(rng);
        for (auto& v : vel)      v = reg_dist(rng) * 0.5f;

        // CPU
        auto cpu_dets = decodeTask(
            hm.data(), center.data(), center_z.data(),
            dim.data(), rot.data(), vel.data(),
            H, W, nc, t, CLS_OFFSETS[t],
            SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE);

        // CUDA：上传 6 个 head tensor
        float *d_hm, *d_center, *d_center_z, *d_dim, *d_rot, *d_vel;
        cudaMalloc(&d_hm,       hm.size()       * sizeof(float));
        cudaMalloc(&d_center,   center.size()   * sizeof(float));
        cudaMalloc(&d_center_z, center_z.size() * sizeof(float));
        cudaMalloc(&d_dim,      dim.size()      * sizeof(float));
        cudaMalloc(&d_rot,      rot.size()      * sizeof(float));
        cudaMalloc(&d_vel,      vel.size()      * sizeof(float));
        cudaMemcpy(d_hm,       hm.data(),       hm.size()       * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_center,   center.data(),   center.size()   * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_center_z, center_z.data(), center_z.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dim,      dim.data(),      dim.size()      * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rot,      rot.data(),      rot.size()      * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,      vel.data(),      vel.size()      * sizeof(float), cudaMemcpyHostToDevice);

        launchDecodeTaskCuda(
            d_hm, d_center, d_center_z, d_dim, d_rot, d_vel,
            H, W, nc, t, CLS_OFFSETS[t], MAX_CANDS_PER_TASK,
            SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE,
            d_cand_buffer + (size_t)t * MAX_CANDS_PER_TASK,
            d_cand_count + t,
            stream);
        cudaStreamSynchronize(stream);

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[task %d] kernel error: %s\n", t, cudaGetErrorString(err));
            return 2;
        }

        unsigned int gpu_K = 0;
        cudaMemcpy(&gpu_K, d_cand_count + t, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (gpu_K > (unsigned int)MAX_CANDS_PER_TASK) {
            fprintf(stderr, "[FAIL task %d] candidate overflow: gpu_K=%u > max=%d\n",
                    t, gpu_K, MAX_CANDS_PER_TASK);
            return 3;
        }
        std::vector<Detection> gpu_dets(gpu_K);
        if (gpu_K > 0) {
            cudaMemcpy(gpu_dets.data(),
                       d_cand_buffer + (size_t)t * MAX_CANDS_PER_TASK,
                       (size_t)gpu_K * sizeof(Detection),
                       cudaMemcpyDeviceToHost);
        }

        cudaFree(d_hm); cudaFree(d_center); cudaFree(d_center_z);
        cudaFree(d_dim); cudaFree(d_rot); cudaFree(d_vel);

        printf("[task %d] cls=%d  cpu=%zu  cuda=%u\n", t, nc, cpu_dets.size(), gpu_K);

        if (cpu_dets.size() != gpu_dets.size()) {
            fprintf(stderr, "[FAIL task %d] count mismatch cpu=%zu cuda=%u\n",
                    t, cpu_dets.size(), gpu_K);
            fail_count++;
            continue;
        }

        std::sort(cpu_dets.begin(), cpu_dets.end(), DetKey::less);
        std::sort(gpu_dets.begin(), gpu_dets.end(), DetKey::less);

        float task_max_err = 0.f;
        int   field_diff = 0;
        for (size_t i = 0; i < cpu_dets.size(); i++) {
            const auto& a = cpu_dets[i];
            const auto& b = gpu_dets[i];
            if (a.cls_id != b.cls_id || a.task_id != b.task_id) {
                fprintf(stderr, "[FAIL task %d idx %zu] cls/task mismatch (%d,%d) vs (%d,%d)\n",
                        t, i, a.cls_id, a.task_id, b.cls_id, b.task_id);
                fail_count++;
                break;
            }
            float fields[10] = {
                std::fabs(a.x - b.x),     std::fabs(a.y - b.y),     std::fabs(a.z - b.z),
                std::fabs(a.w - b.w),     std::fabs(a.l - b.l),     std::fabs(a.h - b.h),
                std::fabs(a.rot - b.rot), std::fabs(a.vx - b.vx),   std::fabs(a.vy - b.vy),
                std::fabs(a.score - b.score)
            };
            for (float e : fields) {
                if (e > task_max_err) task_max_err = e;
                if (e > TOL) field_diff++;
            }
        }
        if (task_max_err > global_max_err) global_max_err = task_max_err;

        printf("           max_abs_err=%.2e  field_diff=%d (tol=%.0e)\n",
               task_max_err, field_diff, TOL);

        if (field_diff > 0) {
            fprintf(stderr, "[FAIL task %d] %d fields exceeded tol\n", t, field_diff);
            fail_count++;
        }

        total_cpu  += (int)cpu_dets.size();
        total_cuda += (int)gpu_K;
    }

    cudaFree(d_cand_buffer);
    cudaFree(d_cand_count);
    cudaStreamDestroy(stream);

    printf("\n[summary] cpu_total=%d  cuda_total=%d  global_max_err=%.2e\n",
           total_cpu, total_cuda, global_max_err);

    if (fail_count > 0) {
        fprintf(stderr, "[FAIL] %d task(s) failed\n", fail_count);
        return 1;
    }
    printf("[OK] CPU 与 CUDA decode + peak filter 输出等价\n");
    return 0;
}
