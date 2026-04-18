#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>

#include <memory>

// 共享 engine：runtime + ICudaEngine 一份，可被多 InferContext 共享
// （ICudaEngine 是 weights+plan，TRT 文档明确 thread-safe，可跨 stream/线程共享）
class InferEngine {
public:
    explicit InferEngine(const std::string& engine_path);
    ~InferEngine();

    nvinfer1::ICudaEngine* engine() const { return engine_.get(); }
    int getNbIOTensors() const;

private:
    class Logger : public nvinfer1::ILogger
    {
        void log(Severity s, const char* msg) noexcept override {
            if (s <= Severity::kWARNING)
                fprintf(stderr, "[TRT] %s\n", msg);
        }
    };

    Logger                                      logger_;
    std::unique_ptr<nvinfer1::IRuntime>         runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>      engine_;
};

// 推理上下文：持有 IExecutionContext + per-binding device buffer + stream。
// 必须由外部 InferEngine 创建——这与 TRT 原生 API 同构
// （IExecutionContext 永远由 ICudaEngine::createExecutionContext() 产生）。
class InferContext {
public:
    explicit InferContext(const InferEngine& engine);
    ~InferContext();

    // 设置动态输入的实际形状（静态输入不需要调用）
    void setInputShape(const std::string& name, const std::vector<int>& shape);

    // 把 CPU 数据拷到 GPU input buffer
    void setInput(const std::string& name, const void* data, size_t bytes);
    void setInputAsync(const std::string& name, const void* data, size_t bytes);

    // 执行推理（infer 会同步当前 stream，inferAsync 只入队）
    void infer();
    void inferAsync();

    // 把 GPU output buffer 拷回 CPU
    void getOutput(const std::string& name, void* data, size_t bytes);
    void getOutputAsync(const std::string& name, void* data, size_t bytes);

    // 让外部 pipeline 统一管理 stream；调用后 InferContext 不再销毁该 stream
    void setStream(cudaStream_t stream);
    cudaStream_t stream() const;

    // 查询 output 的实际 shape（动态输入时推断后才准确）
    std::vector<int> getOutputShape(const std::string& name);

    // 返回 tensor 的 GPU 指针，供外部 kernel 零拷贝读写
    void* getDeviceBuffer(const std::string& name) const;

private:
    nvinfer1::ICudaEngine*                       engine_ = nullptr;   // borrowed
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                                 stream_;
    bool                                         owns_stream_ = true;

    // tensor 名 → GPU buffer 指针
    std::unordered_map<std::string, void*>       buffers_;

    void allocateBuffers();
};
