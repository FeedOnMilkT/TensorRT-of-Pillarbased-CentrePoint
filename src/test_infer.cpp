#include "infer_context.h"
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    // ── 测试 backbone_head（静态输入）──────────────────────────────────────────
    {
        InferContext ctx("/workspace/engines/backbone_head_fp16.plan");

        // dummy input：全零，shape [1, 64, 512, 512]
        std::vector<float> input(1 * 64 * 512 * 512, 0.0f);
        ctx.setInput("spatial_features", input.data(), input.size() * sizeof(float));
        ctx.infer();

        // 读第一个输出 task0_hm，打印 shape 和前几个值
        auto shape = ctx.getOutputShape("task0_hm");
        std::cout << "task0_hm shape: [";
        for (int i = 0; i < (int)shape.size(); ++i)
            std::cout << shape[i] << (i+1<(int)shape.size() ? "," : "");
        std::cout << "]\n";

        size_t n = 1;
        for (int s : shape) n *= s;
        std::vector<float> out(n);
        ctx.getOutput("task0_hm", out.data(), n * sizeof(float));
        std::cout << "task0_hm 前5值: ";
        for (int i = 0; i < 5; ++i) std::cout << out[i] << " ";
        std::cout << "\n";
    }

    // ── 测试 PFN（动态输入）───────────────────────────────────────────────────
    {
        int P = 12000, N = 20, C = 11;
        InferContext ctx("/workspace/engines/pfn_fp16.plan");

        ctx.setInputShape("pillar_features", {P, N, C});
        ctx.setInputShape("pillar_mask",     {P, N});

        std::vector<float> feats(P * N * C, 0.1f);
        std::vector<float> mask(P * N, 1.0f);
        ctx.setInput("pillar_features", feats.data(), feats.size() * sizeof(float));
        ctx.setInput("pillar_mask",     mask.data(),  mask.size()  * sizeof(float));
        ctx.infer();

        auto shape = ctx.getOutputShape("pillar_embeddings");
        std::cout << "pillar_embeddings shape: [" << shape[0] << "," << shape[1] << "]\n";

        std::vector<float> out(P * 64);
        ctx.getOutput("pillar_embeddings", out.data(), out.size() * sizeof(float));
        std::cout << "pillar_embeddings[0] 前5值: ";
        for (int i = 0; i < 5; ++i) std::cout << out[i] << " ";
        std::cout << "\n";
    }

    std::cout << "推理测试完成\n";
    return 0;
}
