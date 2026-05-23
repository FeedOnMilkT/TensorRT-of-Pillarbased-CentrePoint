#include "engine_builder.h"
#include "scatter_plugin.h"
#include "calibrator.h"
#include "project_paths.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// 构建模式：
//   无参数：FP16 构建 backbone+head / PFN / e2e 三个 engine（保持原行为）
//   --int8-pfn <calib_dir>：仅构建 PFN INT8，从 calib_dir 读 dump 数据
//   --int8-bb  <calib_dir>：仅构建 backbone+head INT8
//   --int8-e2e <calib_dir>：仅构建 e2e INT8，从 calib_dir 读三路 dump 数据
//
// 路径解析：onnx / engines / calib cache 路径由 project_paths.h 统一从
// $CENTERPOINT_ROOT（或可执行文件上溯）派生，不再硬编码 /workspace/...。
int main(int argc, char* argv[])
{
    centerpoint_trt::registerPillarScatterPlugin();

    std::string int8_pfn_dir, int8_bb_dir, int8_e2e_dir;
    int calib_frames = 0;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--int8-pfn") == 0 && i + 1 < argc) {
            int8_pfn_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--int8-bb") == 0 && i + 1 < argc) {
            int8_bb_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--int8-e2e") == 0 && i + 1 < argc) {
            int8_e2e_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--calib-frames") == 0 && i + 1 < argc) {
            calib_frames = std::stoi(argv[++i]);
        }
    }

    // ── INT8 PFN ─────────────────────────────────────────────────────────────
    if (!int8_pfn_dir.empty()) {
        int n = calib_frames > 0 ? calib_frames : 32;
        // OPT_P=12000, MAX_PTS=20, C_IN=11
        constexpr size_t FEAT_BYTES = (size_t)12000 * 20 * 11 * sizeof(float);
        constexpr size_t MASK_BYTES = (size_t)12000 * 20 * sizeof(float);
        std::vector<CalibInput> ins = {
            {"pillar_features", "pillar_features", FEAT_BYTES},
            {"pillar_mask",     "pillar_mask",     MASK_BYTES},
        };
        CenterPointCalibrator calib(ins, int8_pfn_dir, n,
                                    centerpoint_trt::enginePath("calib_pfn.cache"));
        bool ok = buildEngine(
            centerpoint_trt::onnxPath("pfn.onnx"),
            centerpoint_trt::enginePath("pfn_int8.plan"),
            /*fp16=*/false,  // INT8 模式下 buildEngine 内部会同时打开 FP16 fallback
            /*workspace_gb=*/2,
            {
                {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
                {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
            },
            &calib);
        if (!ok) { std::cerr << "PFN INT8 构建失败\n"; return 1; }
        std::cout << "PFN INT8 构建完成\n";
        return 0;
    }

    // ── INT8 backbone+head ───────────────────────────────────────────────────
    if (!int8_bb_dir.empty()) {
        int n = calib_frames > 0 ? calib_frames : 32;
        constexpr size_t BB_BYTES = (size_t)1 * 64 * 512 * 512 * sizeof(float);
        std::vector<CalibInput> ins = {
            {"spatial_features", "spatial_features", BB_BYTES},
        };
        CenterPointCalibrator calib(ins, int8_bb_dir, n,
                                    centerpoint_trt::enginePath("calib_bb.cache"));
        bool ok = buildEngine(
            centerpoint_trt::onnxPath("backbone_head.onnx"),
            centerpoint_trt::enginePath("backbone_head_int8.plan"),
            /*fp16=*/false,
            /*workspace_gb=*/2,
            /*profiles=*/{},
            &calib);
        if (!ok) { std::cerr << "backbone+head INT8 构建失败\n"; return 1; }
        std::cout << "backbone+head INT8 构建完成\n";
        return 0;
    }

    // ── INT8 e2e (PFN + scatter plugin + backbone+head) ─────────────────────
    if (!int8_e2e_dir.empty()) {
        int n = calib_frames > 0 ? calib_frames : 32;
        constexpr size_t FEAT_BYTES  = (size_t)12000 * 20 * 11 * sizeof(float);
        constexpr size_t MASK_BYTES  = (size_t)12000 * 20 * sizeof(float);
        constexpr size_t COORD_BYTES = (size_t)12000 * 4 * sizeof(int);
        std::vector<CalibInput> ins = {
            {"pillar_features", "pillar_features", FEAT_BYTES},
            {"pillar_mask",     "pillar_mask",     MASK_BYTES},
            {"pillar_coords",   "pillar_coords",   COORD_BYTES},
        };
        CenterPointCalibrator calib(ins, int8_e2e_dir, n,
                                    centerpoint_trt::enginePath("calib_e2e.cache"));
        bool ok = buildEngine(
            centerpoint_trt::onnxPath("centerpoint_e2e.onnx"),
            centerpoint_trt::enginePath("centerpoint_e2e_int8.plan"),
            /*fp16=*/false,
            /*workspace_gb=*/2,
            {
                {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
                {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
                {"pillar_coords",   {1000,4},     {12000,4},     {30000,4}},
            },
            &calib);
        if (!ok) { std::cerr << "e2e INT8 构建失败\n"; return 1; }
        std::cout << "e2e INT8 构建完成\n";
        return 0;
    }

    // ── 默认 FP16 构建（无 INT8 标志时的原行为） ────────────────────────────
    bool ok = buildEngine(
        centerpoint_trt::onnxPath("backbone_head.onnx"),
        centerpoint_trt::enginePath("backbone_head_fp16.plan"),
        /*fp16=*/true,
        /*workspace_gb=*/2,
        /*profiles=*/{}
    );
    if (!ok) { std::cerr << "backbone engine 构建失败\n"; return 1; }

    ok = buildEngine(
        centerpoint_trt::onnxPath("pfn.onnx"),
        centerpoint_trt::enginePath("pfn_fp16.plan"),
        /*fp16=*/true,
        /*workspace_gb=*/2,
        {
            {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
            {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
        }
    );
    if (!ok) { std::cerr << "PFN engine 构建失败\n"; return 1; }

    ok = buildEngine(
        centerpoint_trt::onnxPath("centerpoint_e2e.onnx"),
        centerpoint_trt::enginePath("centerpoint_e2e_fp16.plan"),
        /*fp16=*/true,
        /*workspace_gb=*/2,
        {
            {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
            {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
            {"pillar_coords",   {1000,4},     {12000,4},     {30000,4}},
        }
    );
    if (!ok) { std::cerr << "端到端 engine 构建失败\n"; return 1; }

    std::cout << "全部 engine 构建完成\n";
    return 0;
}
