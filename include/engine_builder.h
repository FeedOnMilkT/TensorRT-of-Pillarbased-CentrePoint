#pragma once
#include <string>
#include <vector>

struct DynamicShapeProfile {
    std::string input_name;
    std::vector<int> min_shape;
    std::vector<int> opt_shape;
    std::vector<int> max_shape;
};

// 构建并序列化 TRT engine
// profiles 为空时按静态形状构建（backbone），非空时设置 Optimization Profile（PFN）
bool buildEngine(const std::string& onnx_path,
                 const std::string& engine_path,
                 bool fp16 = true,
                 size_t workspace_gb = 2,
                 const std::vector<DynamicShapeProfile>& profiles = {});
