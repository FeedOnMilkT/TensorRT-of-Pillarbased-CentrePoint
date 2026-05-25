#include "engine_builder.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// ── Logger ────────────────────────────────────────────────────────────────────

class TRTLogger : public nvinfer1::ILogger 
{
    public:
        void log(Severity severity, const char* msg) noexcept override 
        {
            if (severity <= Severity::kWARNING)
                std::cerr << "[TRT] " << msg << std::endl;
        }
};

// ── Helper：把 std::vector<int> 转成 nvinfer1::Dims ──────────────────────────

static nvinfer1::Dims toDims(const std::vector<int>& v) 
{
    nvinfer1::Dims d;
    d.nbDims = static_cast<int>(v.size());
    for (int i = 0; i < d.nbDims; ++i) d.d[i] = v[i];
    return d;
}

// ── buildEngine ───────────────────────────────────────────────────────────────

bool buildEngine(const std::string& onnx_path,
                 const std::string& engine_path,
                 bool fp16,
                 size_t workspace_gb,
                 const std::vector<DynamicShapeProfile>& profiles,
                 nvinfer1::IInt8Calibrator* calibrator,
                 float plugin_output_range,
                 const std::string& force_fp16_substr)
{
    TRTLogger logger;

    // 1. Builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));

    if (!builder) 
    { 
        std::cerr << "createInferBuilder failed\n"; return false; 
    }

    // 2. Network（TRT 10 默认 explicit batch，传 0 即可）
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));

    // 3. 解析 ONNX
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    {
        std::ifstream f(onnx_path, std::ios::binary);
        if (!f) 
        { 
            std::cerr << "无法打开 ONNX 文件: " << onnx_path << "\n"; return false; 
        }

        std::vector<char> buf(std::istreambuf_iterator<char>(f), {});

        if (!parser->parse(buf.data(), buf.size())) 
        {
            std::cerr << "ONNX 解析失败\n";
            return false;
        }
    }

    std::cout << "ONNX 解析完成，输入数: " << network->getNbInputs()
              << "，输出数: " << network->getNbOutputs() << "\n";

    // 4. BuilderConfig
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               workspace_gb << 30);
    if (fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 已启用\n";
    }
    if (calibrator) {
        // INT8 主精度；同时打开 FP16 让无法 INT8 化的层 fallback，避免 FP32 兜底
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setInt8Calibrator(calibrator);
        std::cout << "INT8 已启用（FP16 fallback）\n";
        if (!force_fp16_substr.empty()) {
            // 配合 setPrecision 让 hint 真正生效（PREFER 允许 fallback if不可达）
            config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        }
    }

    // 5. Optimization Profile（仅动态输入时需要）
    if (!profiles.empty()) 
    {
        auto profile = builder->createOptimizationProfile();
        for (const auto& p : profiles) {
            profile->setDimensions(p.input_name.c_str(),
                nvinfer1::OptProfileSelector::kMIN, toDims(p.min_shape));
            profile->setDimensions(p.input_name.c_str(),
                nvinfer1::OptProfileSelector::kOPT, toDims(p.opt_shape));
            profile->setDimensions(p.input_name.c_str(),
                nvinfer1::OptProfileSelector::kMAX, toDims(p.max_shape));
        }
        config->addOptimizationProfile(profile);
        std::cout << "Optimization Profile 已设置（" << profiles.size() << " 个输入）\n";
    }

    // 5.5  INT8 plugin output dynamic range workaround
    //
    // TRT 10 在 INT8 build 时，对 IPluginV3 layer 的输出张量不会自动通过 calibrator
    // 收集 quant range（calibrator 只看 engine input + ONNX 内的非 plugin tensor）。
    // 没有 range → TRT 给 default → 下游 INT8 conv 把 plugin 输出 clip 到几乎 0
    // → activation 雪崩 → detection head 全部输出 < threshold → 0 检测。
    //
    // workaround：调用方传入经验 range（spatial_features 在 nuScenes 上 max abs ~6），
    // 这里给所有 plugin layer 的所有 output ITensor 显式 setDynamicRange(-r, +r)。
    // 仅 INT8 模式生效 —— FP16 时 range 没意义。
    if (calibrator && plugin_output_range > 0.0f) {
        int pluginsTouched = 0;
        int tensorsTouched = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* layer = network->getLayer(i);
            if (layer->getType() != nvinfer1::LayerType::kPLUGIN_V3) continue;
            pluginsTouched++;
            for (int o = 0; o < layer->getNbOutputs(); ++o) {
                auto* t = layer->getOutput(o);
                if (!t) continue;
                if (!t->setDynamicRange(-plugin_output_range, plugin_output_range)) {
                    std::cerr << "[engine_builder] WARN: setDynamicRange 失败 on "
                              << layer->getName() << " out[" << o << "]\n";
                } else {
                    tensorsTouched++;
                }
            }
        }
        std::cout << "INT8 plugin output range workaround: 给 "
                  << pluginsTouched << " 个 PluginV3 layer 的 "
                  << tensorsTouched << " 个 output 设了 ["
                  << -plugin_output_range << ", " << plugin_output_range << "]\n";
    }

    // 5.6  INT8 force-FP16 layer-name workaround
    //
    // e2e INT8 整 graph 校准时，因 calibration data 在 OPT padded shape 上 (P=12000，
    // 大量 zero pillar)，PillarScatter 把 padding 累积到 grid[0,0]，backbone 收到的
    // activation 分布跟真实推理 (P~8000) 偏差大；累积到 head 时 quant scale 算偏，
    // 真实推理时 head logit 被压扁 (sigmoid max ~0.05) → 0 检测。
    //
    // workaround：把 head conv (name 含 force_fp16_substr) 强制 FP16，绕开 head
    // 量化。Backbone 仍 INT8，保留主要加速。需要 kPREFER_PRECISION_CONSTRAINTS
    // (上面已设)。
    if (calibrator && !force_fp16_substr.empty()) {
        int touched = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* layer = network->getLayer(i);
            const char* name_c = layer->getName();
            if (!name_c) continue;
            std::string name(name_c);
            if (name.find(force_fp16_substr) == std::string::npos) continue;
            layer->setPrecision(nvinfer1::DataType::kHALF);
            for (int o = 0; o < layer->getNbOutputs(); ++o) {
                layer->setOutputType(o, nvinfer1::DataType::kHALF);
            }
            touched++;
        }
        std::cout << "INT8 force-FP16 layer workaround: "
                  << touched << " 个 layer (name contains \""
                  << force_fp16_substr << "\") 已设 setPrecision(kHALF)\n";
    }

    // 6. 构建并序列化
    std::cout << "开始构建 engine，请等待...\n";
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) 
    { 
        std::cerr << "engine 构建失败\n"; 
        return false; 
    }

    // 7. 写入磁盘
    std::ofstream out(engine_path, std::ios::binary);

    if (!out) 
    { 
        std::cerr << "无法写入: " << engine_path << "\n"; 
        return false; 
    }
    
    out.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cout << "engine 已保存: " << engine_path
              << "（" << serialized->size() / 1024 / 1024 << " MB）\n";
    return true;
}
