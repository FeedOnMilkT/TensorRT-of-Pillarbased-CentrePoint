#include "engine_builder.h"
#include "scatter_plugin.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    // 进程启动时把 PillarScatter creator 注册进 PluginRegistry。
    // 即使本次只构建 PFN/backbone（不含 plugin），多注册一次也无副作用。
    centerpoint_trt::registerPillarScatterPlugin();

    // backbone+head（静态输入）
    bool ok = buildEngine(
        "/workspace/onnx/backbone_head.onnx",
        "/workspace/engines/backbone_head_fp16.plan",
        /*fp16=*/true,
        /*workspace_gb=*/2,
        /*profiles=*/{}
    );
    if (!ok)
    {
        std::cerr << "backbone engine 构建失败\n"; return 1;
    }

    // PFN（动态 pillar 数量，来自 pillar 统计结果）
    ok = buildEngine(
        "/workspace/onnx/pfn.onnx",
        "/workspace/engines/pfn_fp16.plan",
        /*fp16=*/true,
        /*workspace_gb=*/2,
        {
            {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
            {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
        }
    );

    if (!ok)
    {
        std::cerr << "PFN engine 构建失败\n"; return 1;
    }

    // 端到端单引擎（含 PillarScatter plugin）
    ok = buildEngine(
        "/workspace/onnx/centerpoint_e2e.onnx",
        "/workspace/engines/centerpoint_e2e_fp16.plan",
        /*fp16=*/true,
        /*workspace_gb=*/2,
        {
            {"pillar_features", {1000,20,11}, {12000,20,11}, {30000,20,11}},
            {"pillar_mask",     {1000,20},    {12000,20},    {30000,20}},
            {"pillar_coords",   {1000,4},     {12000,4},     {30000,4}},
        }
    );
    if (!ok)
    {
        std::cerr << "端到端 engine 构建失败\n"; return 1;
    }

    std::cout << "全部 engine 构建完成\n";
    return 0;
}
