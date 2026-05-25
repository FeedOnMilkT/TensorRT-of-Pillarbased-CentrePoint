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
//
// plugin_output_range > 0 时，遍历 network 找所有 IPluginV3 layer，给每个
// output ITensor 调用 setDynamicRange(-r, +r)。用途：解决 TRT 10 INT8 build 时
// plugin 输出（如 PillarScatter 的 spatial_features）没有 calibrated quant range
// 导致下游 INT8 conv 把激活全 clip 成 0 的问题。仅 INT8 模式生效，FP16 build 忽略。
//
// force_fp16_substr 非空时，遍历 network 把 name 含该 substr 的 layer 强制
// setPrecision(kHALF) + setOutputType(kHALF)，并开启 kPREFER_PRECISION_CONSTRAINTS。
// 用途：解决 e2e INT8 build 时 head Conv 因 graph-wide 量化误差累积导致 logit
// 动态范围被压扁 (sigmoid 输出 max 仅 ~0.05) → 0 检测的问题。
// CenterPoint 用 "heads_list" 即可一次命中全部 36 个 head conv。
// 仅 INT8 模式生效（FP16 时所有层本来就是 FP16/FP32，无意义）。
bool buildEngine(const std::string& onnx_path,
                 const std::string& engine_path,
                 bool fp16 = true,
                 size_t workspace_gb = 2,
                 const std::vector<DynamicShapeProfile>& profiles = {},
                 nvinfer1::IInt8Calibrator* calibrator = nullptr,
                 float plugin_output_range = 0.0f,
                 const std::string& force_fp16_substr = "");
