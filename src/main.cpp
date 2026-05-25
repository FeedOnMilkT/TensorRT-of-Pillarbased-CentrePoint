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
//   --fp16-e2e：仅构 centerpoint_e2e_fp16
//   --fp16-4a ：仅构 pfn_fp16 + backbone_head_fp16
//   --int8-4a <calib_dir>：一个进程内顺序构 pfn_int8 + backbone_head_int8（calibrator 复用同一 calib_dir）
//
// 路径解析：onnx / engines / calib cache 路径由 project_paths.h 统一从
// $CENTERPOINT_ROOT（或可执行文件上溯）派生，不再硬编码 /workspace/...。
int main(int argc, char* argv[])
{
    centerpoint_trt::registerPillarScatterPlugin();

    std::string int8_pfn_dir, int8_bb_dir, int8_e2e_dir;
    int calib_frames = 0;
    bool fp16_e2e_only = false;
    bool fp16_4a_only  = false;
    std::string int8_4a_dir;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--int8-pfn") == 0 && i + 1 < argc) {
            int8_pfn_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--int8-bb") == 0 && i + 1 < argc) {
            int8_bb_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--int8-e2e") == 0 && i + 1 < argc) {
            int8_e2e_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--calib-frames") == 0 && i + 1 < argc) {
            calib_frames = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--fp16-e2e") == 0) {
            fp16_e2e_only = true;
        } else if (std::strcmp(argv[i], "--fp16-4a") == 0) {
            fp16_4a_only = true;
        } else if (std::strcmp(argv[i], "--int8-4a") == 0 && i + 1 < argc) {
            int8_4a_dir = argv[++i];
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

    // ── INT8 4a 聚合（pfn + backbone_head 同进程顺序构，共用一个 calib_dir） ──
    if (!int8_4a_dir.empty()) {
        int n = calib_frames > 0 ? calib_frames : 32;
        // (1) PFN INT8（与 --int8-pfn 分支等价）
        {
            constexpr size_t FEAT_BYTES = (size_t)12000 * 20 * 11 * sizeof(float);
            constexpr size_t MASK_BYTES = (size_t)12000 * 20 * sizeof(float);
            std::vector<CalibInput> ins = {
                {"pillar_features", "pillar_features", FEAT_BYTES},
                {"pillar_mask",     "pillar_mask",     MASK_BYTES},
            };
            CenterPointCalibrator calib(ins, int8_4a_dir, n,
                                        centerpoint_trt::enginePath("calib_pfn.cache"));
            bool ok = buildEngine(
                centerpoint_trt::onnxPath("pfn.onnx"),
                centerpoint_trt::enginePath("pfn_int8.plan"),
                /*fp16=*/false,
                /*workspace_gb=*/2,
                {
                    {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
                    {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
                },
                &calib);
            if (!ok) { std::cerr << "PFN INT8 (4a-pair) 构建失败\n"; return 1; }
            std::cout << "PFN INT8 (4a-pair) 构建完成\n";
        }
        // (2) backbone+head INT8（与 --int8-bb 分支等价）
        {
            constexpr size_t BB_BYTES = (size_t)1 * 64 * 512 * 512 * sizeof(float);
            std::vector<CalibInput> ins = {
                {"spatial_features", "spatial_features", BB_BYTES},
            };
            CenterPointCalibrator calib(ins, int8_4a_dir, n,
                                        centerpoint_trt::enginePath("calib_bb.cache"));
            bool ok = buildEngine(
                centerpoint_trt::onnxPath("backbone_head.onnx"),
                centerpoint_trt::enginePath("backbone_head_int8.plan"),
                /*fp16=*/false,
                /*workspace_gb=*/2,
                /*profiles=*/{},
                &calib);
            if (!ok) { std::cerr << "backbone+head INT8 (4a-pair) 构建失败\n"; return 1; }
            std::cout << "backbone+head INT8 (4a-pair) 构建完成\n";
        }
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
        // PillarScatter plugin 输出 spatial_features 经 calib_data 抽样 500 帧统计：
        // max abs ≈ 5.9，p99.9 abs ≈ 2.9。取 6.0 略保守覆盖 outlier。
        // 见 engine_builder.cpp 5.5 节注释：TRT 10 INT8 不给 plugin output collect
        // range，没这一行 e2e_int8 会出 0 检测。
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
            &calib,
            /*plugin_output_range=*/6.0f,
            // force head conv FP16. e2e INT8 graph 校准时 padded P=12000 让 PillarScatter
            // 在 grid[0,0] 累积 padding artifact → backbone activation 分布漂移 → head
            // quant scale 算偏 → 真实推理 logit 压扁。让 head FP16 绕开（backbone 仍 INT8）。
            // ONNX node name 形如 bb_/heads_list.{0..5}/{hm,center,...}/.1/Conv，
            // 子串 "heads_list" 一次命中全部 36 个 head conv + 它们之前的 conv-bn-relu 块。
            /*force_fp16_substr=*/"heads_list");
        if (!ok) { std::cerr << "e2e INT8 构建失败\n"; return 1; }
        std::cout << "e2e INT8 构建完成\n";
        return 0;
    }

    // ── 默认 FP16 构建（带 --fp16-X 时仅构指定的子集；不带 = 三件套全构，向后兼容）
    bool any_fp16_flag = fp16_e2e_only || fp16_4a_only;
    bool build_4a_fp16  = !any_fp16_flag || fp16_4a_only;
    bool build_e2e_fp16 = !any_fp16_flag || fp16_e2e_only;

    if (build_4a_fp16) {
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
    }

    if (build_e2e_fp16) {
        bool ok = buildEngine(
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
    }

    std::cout << "FP16 构建完成 (4a=" << build_4a_fp16 << " e2e=" << build_e2e_fp16 << ")\n";
    return 0;
}
