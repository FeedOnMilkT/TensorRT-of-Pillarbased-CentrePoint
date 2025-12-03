#pragma once
#include <string>
#include <vector>

namespace nvinfer1 { class IInt8Calibrator; }

struct DynamicShapeProfile {
    std::string input_name;
    std::vector<int> min_shape;
    std::vector<int> opt_shape;
    std::vector<int> max_shape;
};

// 构建并序列化 TRT engine
// profiles 为空时按静态形状构建（backbone），非空时设置 Optimization Profile（PFN）
// calibrator 非空时启用 INT8（同时 fp16 仍然作为 fallback 精度，所以会一起设上）
bool buildEngine(const std::string& onnx_path,
                 const std::string& engine_path,
                 bool fp16 = true,
                 size_t workspace_gb = 2,
                 const std::vector<DynamicShapeProfile>& profiles = {},
                 nvinfer1::IInt8Calibrator* calibrator = nullptr);
