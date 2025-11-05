// PillarScatter plugin 单测
// 覆盖：
//   1. 直接调 plugin.enqueue，与 4a 直接调 launchPillarScatter 数值完全一致
//   2. clone 后的 plugin 行为与原 plugin 一致
//   3. getFieldsToSerialize -> creator.createPlugin round-trip：H/W/C 还原后行为不变

#include "scatter_plugin.h"
#include "scatter_kernel.h"

#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <vector>

using namespace centerpoint_trt;
using namespace nvinfer1;

namespace {

constexpr int kH = 8;
constexpr int kW = 8;
constexpr int kC = 4;
constexpr int kP = 3;

// ── 测试夹具：构造 [3 个 pillar] 输入，写到独立的 device 缓冲区 ─────────────
struct DeviceFixture
{
    float* d_features = nullptr;
    int*   d_coords   = nullptr;
    float* d_canvas   = nullptr;

    DeviceFixture()
    {
        cudaMalloc(&d_features, kP * kC * sizeof(float));
        cudaMalloc(&d_coords,   kP * 4  * sizeof(int));
        cudaMalloc(&d_canvas,   kC * kH * kW * sizeof(float));

        // pillar 0: 坐标 (0, 2, 3)，特征全 1.0
        // pillar 1: 坐标 (0, 5, 1)，特征 [10,20,30,40]
        // pillar 2: 坐标 (0, 0, 7)，特征全 -1.0
        std::vector<float> features = {
            1.0f, 1.0f, 1.0f, 1.0f,
            10.0f, 20.0f, 30.0f, 40.0f,
            -1.0f, -1.0f, -1.0f, -1.0f,
        };
        std::vector<int> coords = {
            0, 0, 2, 3,
            0, 0, 5, 1,
            0, 0, 0, 7,
        };
        cudaMemcpy(d_features, features.data(), features.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_coords, coords.data(), coords.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    ~DeviceFixture()
    {
        cudaFree(d_features);
        cudaFree(d_coords);
        cudaFree(d_canvas);
    }
};

std::vector<float> readCanvas(float const* d_canvas)
{
    std::vector<float> h(kC * kH * kW);
    cudaMemcpy(h.data(), d_canvas, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// 用 plugin.enqueue 跑一遍 scatter，返回 host 端 canvas
std::vector<float> runViaPlugin(IPluginV3OneRuntime* plugin, DeviceFixture& fix)
{
    cudaMemset(fix.d_canvas, 0, kC * kH * kW * sizeof(float));

    PluginTensorDesc inDesc[2]{};
    inDesc[0].dims = Dims{2, {kP, kC}};
    inDesc[0].type = DataType::kFLOAT;
    inDesc[0].format = TensorFormat::kLINEAR;
    inDesc[1].dims = Dims{2, {kP, 4}};
    inDesc[1].type = DataType::kINT32;
    inDesc[1].format = TensorFormat::kLINEAR;

    PluginTensorDesc outDesc[1]{};
    outDesc[0].dims = Dims{4, {1, kC, kH, kW}};
    outDesc[0].type = DataType::kFLOAT;
    outDesc[0].format = TensorFormat::kLINEAR;

    void const* inputs[2]  = {fix.d_features, fix.d_coords};
    void*       outputs[1] = {fix.d_canvas};

    int rc = plugin->enqueue(inDesc, outDesc, inputs, outputs, nullptr, /*stream=*/0);
    if (rc != 0) {
        std::cerr << "enqueue returned " << rc << std::endl;
        std::exit(1);
    }
    cudaDeviceSynchronize();
    return readCanvas(fix.d_canvas);
}

// 直接调 4a kernel，返回 host 端 canvas（黄金参考）
std::vector<float> runViaKernel(DeviceFixture& fix)
{
    cudaMemset(fix.d_canvas, 0, kC * kH * kW * sizeof(float));
    launchPillarScatter(fix.d_features, fix.d_coords, fix.d_canvas,
                        kP, kC, kH, kW, /*stream=*/0);
    cudaDeviceSynchronize();
    return readCanvas(fix.d_canvas);
}

bool sameVector(std::vector<float> const& a, std::vector<float> const& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

}  // namespace

int main()
{
    registerPillarScatterPlugin();

    DeviceFixture fix;
    auto const golden = runViaKernel(fix);

    int failed = 0;

    // ── 测 1：plugin.enqueue 与 4a kernel 一致 ────────────────────────────
    {
        PillarScatterPlugin plugin(kH, kW, kC);
        auto out = runViaPlugin(&plugin, fix);
        if (!sameVector(out, golden)) {
            std::cerr << "[FAIL] enqueue parity" << std::endl;
            ++failed;
        } else {
            std::cout << "[PASS] enqueue parity with launchPillarScatter" << std::endl;
        }
    }

    // ── 测 2：clone 后行为一致 ─────────────────────────────────────────────
    {
        PillarScatterPlugin original(kH, kW, kC);
        auto* cloned = original.clone();
        auto* runtime = static_cast<IPluginV3OneRuntime*>(
            cloned->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        auto out = runViaPlugin(runtime, fix);
        delete cloned;
        if (!sameVector(out, golden)) {
            std::cerr << "[FAIL] clone parity" << std::endl;
            ++failed;
        } else {
            std::cout << "[PASS] clone parity" << std::endl;
        }
    }

    // ── 测 3：序列化 round-trip（getFieldsToSerialize → creator.createPlugin） ──
    {
        PillarScatterPlugin original(kH, kW, kC);
        auto* fc = original.getFieldsToSerialize();

        PillarScatterPluginCreator creator;
        auto* recreated = creator.createPlugin("PillarScatter", fc, TensorRTPhase::kRUNTIME);
        if (!recreated) {
            std::cerr << "[FAIL] createPlugin returned null" << std::endl;
            ++failed;
        } else {
            auto* runtime = static_cast<IPluginV3OneRuntime*>(
                recreated->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
            auto out = runViaPlugin(runtime, fix);
            delete recreated;
            if (!sameVector(out, golden)) {
                std::cerr << "[FAIL] serialize round-trip parity" << std::endl;
                ++failed;
            } else {
                std::cout << "[PASS] serialize round-trip parity" << std::endl;
            }
        }
    }

    // ── 测 4：creator 拒绝非法 attribute（H=0） ────────────────────────────
    {
        int badH = 0, goodW = kW, goodC = kC;
        PluginField fields[3] = {
            {"H", &badH,  PluginFieldType::kINT32, 1},
            {"W", &goodW, PluginFieldType::kINT32, 1},
            {"C", &goodC, PluginFieldType::kINT32, 1},
        };
        PluginFieldCollection fc{3, fields};

        PillarScatterPluginCreator creator;
        auto* p = creator.createPlugin("PillarScatter", &fc, TensorRTPhase::kBUILD);
        if (p != nullptr) {
            std::cerr << "[FAIL] creator accepted invalid H=0" << std::endl;
            delete p;
            ++failed;
        } else {
            std::cout << "[PASS] creator rejects invalid attributes" << std::endl;
        }
    }

    if (failed == 0) {
        std::cout << "All plugin unit tests passed." << std::endl;
        return 0;
    }
    std::cerr << failed << " test(s) failed." << std::endl;
    return 1;
}
