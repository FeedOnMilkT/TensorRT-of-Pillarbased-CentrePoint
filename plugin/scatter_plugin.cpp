#include "scatter_plugin.h"
#include "scatter_kernel.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

namespace centerpoint_trt {

using namespace nvinfer1;

// ─── PillarScatterPlugin ──────────────────────────────────────────────────────

PillarScatterPlugin::PillarScatterPlugin(int H, int W, int C)
    : mH(H), mW(W), mC(C)
{
}

IPluginCapability* PillarScatterPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    switch (type) {
        case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

IPluginV3* PillarScatterPlugin::clone() noexcept
{
    try {
        return new PillarScatterPlugin(*this);
    } catch (...) {
        return nullptr;
    }
}

int32_t PillarScatterPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,  int32_t nbInputs,
    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // 校验：2 输入 1 输出，shape 维度数符合预期
    if (nbInputs != 2 || nbOutputs != 1) return -1;
    if (in[0].desc.dims.nbDims != 2) return -1;          // [P, C]
    if (in[1].desc.dims.nbDims != 2) return -1;          // [P, 4]
    if (out[0].desc.dims.nbDims != 4) return -1;         // [1, C, H, W]
    return 0;
}

bool PillarScatterPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // pos 0: pillar_embeddings   FLOAT, LINEAR
    // pos 1: pillar_coords       INT32, LINEAR
    // pos 2: canvas（output 0）  FLOAT, LINEAR
    auto const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    if (pos == 0) return d.type == DataType::kFLOAT;
    if (pos == 1) return d.type == DataType::kINT32;
    if (pos == 2) return d.type == DataType::kFLOAT;
    return false;
}

int32_t PillarScatterPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[0] = inputTypes[0];  // canvas 与 pillar_embeddings 同类型
    return 0;
}

int32_t PillarScatterPlugin::getOutputShapes(
    DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    // canvas [1, C, H, W]：完全由 attribute 决定，不依赖输入 P
    outputs[0].nbDims = 4;
    outputs[0].d[0] = exprBuilder.constant(1);
    outputs[0].d[1] = exprBuilder.constant(mC);
    outputs[0].d[2] = exprBuilder.constant(mH);
    outputs[0].d[3] = exprBuilder.constant(mW);
    return 0;
}

int32_t PillarScatterPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs,
    void* /*workspace*/, cudaStream_t stream) noexcept
{
    // 实际 pillar 数 P 来自运行时输入 0 的第 0 维
    int const P = static_cast<int>(inputDesc[0].dims.d[0]);

    auto const* d_features = static_cast<float const*>(inputs[0]);
    auto const* d_coords   = static_cast<int32_t const*>(inputs[1]);
    auto*       d_canvas   = static_cast<float*>(outputs[0]);

    // canvas 必须先清零：未被任何 pillar 覆盖的格子要保持 0
    size_t const canvas_bytes = static_cast<size_t>(mC) * mH * mW * sizeof(float);
    cudaError_t err = cudaMemsetAsync(d_canvas, 0, canvas_bytes, stream);
    if (err != cudaSuccess) return -1;

    if (P > 0) {
        launchPillarScatter(d_features, d_coords, d_canvas, P, mC, mH, mW, stream);
    }
    return 0;
}

PluginFieldCollection const* PillarScatterPlugin::getFieldsToSerialize() noexcept
{
    // 把 H/W/C 打包返回，TRT 在序列化引擎时调用此函数；
    // 反序列化时 creator 的 createPlugin 会拿到同样的 PluginFieldCollection。
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField{"H", &mH, PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(PluginField{"W", &mW, PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(PluginField{"C", &mC, PluginFieldType::kINT32, 1});
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields   = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ─── PillarScatterPluginCreator ───────────────────────────────────────────────

PillarScatterPluginCreator::PillarScatterPluginCreator()
{
    mFields.emplace_back(PluginField{"H", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"W", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"C", nullptr, PluginFieldType::kINT32, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields   = mFields.data();
}

IPluginV3* PillarScatterPluginCreator::createPlugin(
    char const* /*name*/, PluginFieldCollection const* fc,
    TensorRTPhase /*phase*/) noexcept
{
    int H = 0, W = 0, C = 0;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (std::strcmp(f.name, "H") == 0) H = *static_cast<int const*>(f.data);
        else if (std::strcmp(f.name, "W") == 0) W = *static_cast<int const*>(f.data);
        else if (std::strcmp(f.name, "C") == 0) C = *static_cast<int const*>(f.data);
    }
    if (H <= 0 || W <= 0 || C <= 0) {
        std::cerr << "[PillarScatter] invalid attributes H=" << H
                  << " W=" << W << " C=" << C << std::endl;
        return nullptr;
    }
    try {
        return new PillarScatterPlugin(H, W, C);
    } catch (...) {
        return nullptr;
    }
}

// ─── 全局注册 ─────────────────────────────────────────────────────────────────

void registerPillarScatterPlugin()
{
    static PillarScatterPluginCreator sCreator;
    static bool sRegistered = false;
    if (!sRegistered) {
        getPluginRegistry()->registerCreator(sCreator, kSCATTER_PLUGIN_NS);
        sRegistered = true;
    }
}

}  // namespace centerpoint_trt
