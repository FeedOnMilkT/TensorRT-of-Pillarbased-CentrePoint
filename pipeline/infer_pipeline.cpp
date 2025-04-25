#include "infer_context.h"
#include "scatter_kernel.h"
#include "pillarize.h"
#include "postprocess.h"
#include "bench.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

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

// 把一帧的 detections 追加到 JSON 流（调用方管理逗号分隔）
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
    std::string json_path;
    std::string file_list_path;
    std::vector<std::string> bin_files;

    bool benchmark = false;
    int  warmup_iters = 10;
    int  bench_iters = 200;
    std::string bench_output;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--file-list" && i + 1 < argc) {
            file_list_path = argv[++i];
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_iters = std::stoi(argv[++i]);
        } else if (arg == "--bench-iters" && i + 1 < argc) {
            bench_iters = std::stoi(argv[++i]);
        } else if (arg == "--bench-output" && i + 1 < argc) {
            bench_output = argv[++i];
        } else {
            bin_files.push_back(arg);
        }
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
        std::cerr << "Usage: infer_pipeline <file.bin> [...] [--file-list files.txt]\n"
                  << "                       [--json out.json]\n"
                  << "                       [--benchmark [--warmup N] [--bench-iters N] [--bench-output bench.json]]\n";
        return 1;
    }

    InferContext pfn_ctx("/workspace/engines/pfn_fp16.plan");
    InferContext bb_ctx("/workspace/engines/backbone_head_fp16.plan");

    int* d_coords;
    cudaMalloc(&d_coords, MAX_PILLARS * 4 * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 单帧 pipeline：可选传入 timer 做分阶段计时；返回当帧检测框
    auto runOneFrame = [&](const float* pts_ptr, int N, StageTimer* timer)
                       -> std::vector<Detection> {
        if (timer) timer->beginFrameGpu(stream);

        if (timer) timer->begin("pillarize");
        auto batch = pillarize(pts_ptr, N,
            X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
            VX, VY, VZ, MAX_PTS, MAX_PILLARS);
        int P = batch.P;
        if (timer) timer->end("pillarize");

        if (timer) timer->begin("h2d_coords");
        cudaMemcpy(d_coords, batch.coords.data(), P * 4 * sizeof(int),
                   cudaMemcpyHostToDevice);
        if (timer) timer->end("h2d_coords");

        if (timer) timer->begin("pfn");
        pfn_ctx.setInputShape("pillar_features", {P, MAX_PTS, C_IN});
        pfn_ctx.setInputShape("pillar_mask",     {P, MAX_PTS});
        pfn_ctx.setInput("pillar_features",
                         batch.features.data(), P * MAX_PTS * C_IN * sizeof(float));
        pfn_ctx.setInput("pillar_mask",
                         batch.mask.data(),     P * MAX_PTS * sizeof(float));
        pfn_ctx.infer();
        if (timer) timer->end("pfn");

        if (timer) timer->begin("scatter");
        float* d_canvas = static_cast<float*>(bb_ctx.getDeviceBuffer("spatial_features"));
        cudaMemset(d_canvas, 0, C_OUT * BEV_H * BEV_W * sizeof(float));
        const float* d_emb = static_cast<const float*>(
            pfn_ctx.getDeviceBuffer("pillar_embeddings"));
        launchPillarScatter(d_emb, d_coords, d_canvas, P, C_OUT, BEV_H, BEV_W, stream);
        cudaStreamSynchronize(stream);
        if (timer) timer->end("scatter");

        if (timer) timer->begin("backbone");
        bb_ctx.infer();
        if (timer) timer->end("backbone");

        if (timer) timer->begin("decode_nms");
        std::vector<Detection> all_dets;
        int cls_offset = 0;
        for (int t = 0; t < 6; t++) {
            std::string pre = "task" + std::to_string(t) + "_";
            int nc = TASK_NUM_CLS[t];

            auto hm_shape = bb_ctx.getOutputShape(pre + "hm");
            int H = hm_shape[2], W = hm_shape[3];
            size_t map = (size_t)(H * W) * sizeof(float);

            std::vector<float> hm(nc*H*W), center(2*H*W), center_z(H*W);
            std::vector<float> dim(3*H*W), rot(2*H*W), vel(2*H*W);

            bb_ctx.getOutput(pre+"hm",       hm.data(),       nc*map);
            bb_ctx.getOutput(pre+"center",   center.data(),    2*map);
            bb_ctx.getOutput(pre+"center_z", center_z.data(),  map);
            bb_ctx.getOutput(pre+"dim",      dim.data(),       3*map);
            bb_ctx.getOutput(pre+"rot",      rot.data(),       2*map);
            bb_ctx.getOutput(pre+"vel",      vel.data(),       2*map);

            auto dets = decodeTask(
                hm.data(), center.data(), center_z.data(),
                dim.data(), rot.data(), vel.data(),
                H, W, nc, t, cls_offset,
                SCORE_THRESH, X_MIN, Y_MIN, VX, VY, OUT_STRIDE);
            dets = circleNMS(dets, TASK_NMS_RADIUS[t]);
            all_dets.insert(all_dets.end(), dets.begin(), dets.end());
            cls_offset += nc;
        }
        if (timer) timer->end("decode_nms");

        if (timer) timer->endFrameGpu(stream);
        return all_dets;
    };

    if (benchmark) {
        std::cerr << "[benchmark] preloading " << bin_files.size()
                  << " bin files into memory (excluded from timing)...\n";
        std::vector<std::vector<float>> preloaded;
        std::vector<int> preloaded_N;
        preloaded.reserve(bin_files.size());
        preloaded_N.reserve(bin_files.size());
        for (const auto& p : bin_files) {
            int N;
            preloaded.push_back(loadBin(p, N));
            preloaded_N.push_back(N);
        }
        std::cerr << "[benchmark] warmup=" << warmup_iters
                  << "  iters=" << bench_iters << "\n";

        StageTimer timer;
        for (int i = 0; i < warmup_iters; i++) {
            int idx = i % (int)preloaded.size();
            (void)runOneFrame(preloaded[idx].data(), preloaded_N[idx], nullptr);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < bench_iters; i++) {
            int idx = i % (int)preloaded.size();
            (void)runOneFrame(preloaded[idx].data(), preloaded_N[idx], &timer);
            timer.commitFrame();
        }

        timer.summary(std::cout);
        if (!bench_output.empty()) {
            timer.writeJson(bench_output);
            std::cerr << "[benchmark] wrote " << bench_output << "\n";
        }
    } else {
        std::ofstream jf;
        if (!json_path.empty()) {
            jf.open(json_path);
            jf << "{\n";
        }

        int total = (int)bin_files.size();
        for (int fi = 0; fi < total; fi++) {
            const auto& bin_path = bin_files[fi];

            int N;
            auto pts = loadBin(bin_path, N);

            auto all_dets = runOneFrame(pts.data(), N, nullptr);

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
    }

    cudaFree(d_coords);
    cudaStreamDestroy(stream);
    return 0;
}
