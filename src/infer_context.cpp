#include "infer_context.h"
#include <NvInfer.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

// ── 工具：vector<int> → nvinfer1::Dims ───────────────────────────────────────

static nvinfer1::Dims toDims(const std::vector<int>& v) 
{
    nvinfer1::Dims d;
    d.nbDims = static_cast<int>(v.size());
    for (int i = 0; i < d.nbDims; ++i) d.d[i] = v[i];
    return d;
}

static std::vector<int> fromDims(const nvinfer1::Dims& d) 
{
    std::vector<int> v(d.nbDims);
    for (int i = 0; i < d.nbDims; ++i) v[i] = d.d[i];
    return v;
}

static size_t dimsVolume(const nvinfer1::Dims& d) 
{
    size_t vol = 1;
    for (int i = 0; i < d.nbDims; ++i) vol *= d.d[i];
    return vol;
}

// ── 构造：加载 engine ─────────────────────────────────────────────────────────

InferContext::InferContext(const std::string& engine_path) 
{
    // 读取 .plan 文件
    std::ifstream f(engine_path, std::ios::binary);

    if (!f) throw std::runtime_error("无法打开 engine: " + engine_path);

    std::vector<char> buf(std::istreambuf_iterator<char>(f), {});

    // 反序列化
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), buf.size()));

    if (!engine_) throw std::runtime_error("engine 反序列化失败");

    context_.reset(engine_->createExecutionContext());

    if (!context_) throw std::runtime_error("ExecutionContext 创建失败");

    cudaStreamCreate(&stream_);
    allocateBuffers();

    std::cout << "engine 加载完成: " << engine_path
              << "（" << engine_->getNbIOTensors() << " 个 IO tensors）\n";
}

InferContext::~InferContext() 
{
    for (auto& [name, ptr] : buffers_) cudaFree(ptr);
    cudaStreamDestroy(stream_);
}

// ── allocateBuffers：按 max shape 预分配 GPU 内存 ─────────────────────────────

void InferContext::allocateBuffers() {
    int n = engine_->getNbIOTensors();

    // 先把所有动态输入设为 MAX shape，使 context 能推断输出 shape
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims maxDims = engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMAX);
            context_->setInputShape(name, maxDims);
        }
    }

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims;

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            dims = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
        else
            dims = context_->getTensorShape(name);  // 输入设为 MAX 后，输出 shape 已确定

        size_t bytes = dimsVolume(dims) * sizeof(float);
        void* ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        buffers_[name] = ptr;

        context_->setTensorAddress(name, ptr);
    }
}

// ── setInputShape：动态输入设实际形状 ────────────────────────────────────────

void InferContext::setInputShape(const std::string& name,
                                  const std::vector<int>& shape) 
                                  {
    context_->setInputShape(name.c_str(), toDims(shape));
}

// ── setInput：CPU → GPU ───────────────────────────────────────────────────────

void InferContext::setInput(const std::string& name,
                             const void* data, size_t bytes) 
                             {
    cudaMemcpyAsync(buffers_.at(name), data, bytes,
                    cudaMemcpyHostToDevice, stream_);
}

// ── infer ─────────────────────────────────────────────────────────────────────

void InferContext::infer() 
{
    context_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);
}

// ── getOutput：GPU → CPU ──────────────────────────────────────────────────────

void InferContext::getOutput(const std::string& name,
                              void* data, size_t bytes) 
                              {
    cudaMemcpyAsync(data, buffers_.at(name), bytes,
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

// ── getOutputShape ────────────────────────────────────────────────────────────

std::vector<int> InferContext::getOutputShape(const std::string& name) 
{
    return fromDims(context_->getTensorShape(name.c_str()));
}
