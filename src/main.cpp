#include "engine_builder.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) 
{
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
            {"pillar_features", {5706,20,11}, {22474,20,11}, {26930,20,11}},
            {"pillar_mask",     {5706,20},    {22474,20},    {26930,20}},
        }
    );
    
    if (!ok) 
    { 
        std::cerr << "PFN engine 构建失败\n"; return 1; 
    }

    std::cout << "全部 engine 构建完成\n";
    return 0;
}
