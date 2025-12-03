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
                 nvinfer1::IInt8Calibrator* calibrator)
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
