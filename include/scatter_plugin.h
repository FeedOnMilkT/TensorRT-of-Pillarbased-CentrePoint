#pragma once

#include "NvInferPlugin.h"
#include "NvInferRuntimePlugin.h"
#include <string>
#include <vector>

namespace centerpoint_trt {

constexpr char const* kSCATTER_PLUGIN_NAME    = "PillarScatter";
constexpr char const* kSCATTER_PLUGIN_VERSION = "1";
// 注意：plugin 注册到 *空* namespace，因为 TRT ONNX parser 的
// checkFallbackPluginImporter 始终用空 namespace 查 registry。
// ONNX 节点的 domain 仍保留 "centerpoint_trt"（gs 拼接脚本里），用于满足 ONNX
// custom-op 的 opset_import 规范，但 parser 不用它做 plugin 查找。
constexpr char const* kSCATTER_PLUGIN_NS      = "";

// IPluginV3 单类实现：同时承担 OneCore / OneBuild / OneRuntime 三种能力。
// 三种能力通过 getCapabilityInterface 按需分发回 this 指针。
//
// 输入：
//   [0] pillar_embeddings : float [P, C]     PFN 输出
//   [1] pillar_coords     : int32 [P, 4]     (batch, z, y, x)
// 输出：
//   [0] canvas            : float [1, C, H, W]
// Attribute：
//   H, W, C （int32）
class PillarScatterPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime
{
public:
    PillarScatterPlugin(int H, int W, int C);
    ~PillarScatterPlugin() override = default;

    PillarScatterPlugin(PillarScatterPlugin const&) = default;

    // ── IPluginV3 ───────────────────────────────────────────────────────────
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // ── IPluginV3OneCore ────────────────────────────────────────────────────
    char const* getPluginName() const noexcept override    { return kSCATTER_PLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override { return kSCATTER_PLUGIN_VERSION; }
    char const* getPluginNamespace() const noexcept override { return kSCATTER_PLUGIN_NS; }

    // ── IPluginV3OneBuild ───────────────────────────────────────────────────
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,  int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return 0;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override { return 0; }
    int32_t getNbTactics() noexcept override { return 0; }

    // ── IPluginV3OneRuntime ─────────────────────────────────────────────────
    int32_t setTactic(int32_t tactic) noexcept override { return 0; }

    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in,  int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int                            mH{0};
    int                            mW{0};
    int                            mC{0};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection    mFCToSerialize{};
};

// Creator：负责从 PluginFieldCollection 实例化 plugin（构建期 / 反序列化期均走此路径）。
class PillarScatterPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    PillarScatterPluginCreator();
    ~PillarScatterPluginCreator() override = default;

    char const* getPluginName() const noexcept override      { return kSCATTER_PLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override   { return kSCATTER_PLUGIN_VERSION; }
    char const* getPluginNamespace() const noexcept override { return kSCATTER_PLUGIN_NS; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    nvinfer1::IPluginV3* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection    mFC{};
};

// 进程启动时调用一次：把 creator 注册进全局 PluginRegistry。
// 构建/加载 engine 之前必须先注册，否则 onnx parser 找不到自定义算子会报错。
void registerPillarScatterPlugin();

}  // namespace centerpoint_trt
