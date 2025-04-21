#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>

#include <memory>

class InferContext {
public:
    explicit InferContext(const std::string& engine_path);
    ~InferContext();

    // 设置动态输入的实际形状（静态输入不需要调用）
    void setInputShape(const std::string& name, const std::vector<int>& shape);

    // 把 CPU 数据拷到 GPU input buffer
    void setInput(const std::string& name, const void* data, size_t bytes);

    // 执行推理（异步，内部 stream 同步）
    void infer();

    // 把 GPU output buffer 拷回 CPU
    void getOutput(const std::string& name, void* data, size_t bytes);

    // 查询 output 的实际 shape（动态输入时推断后才准确）
    std::vector<int> getOutputShape(const std::string& name);

    // 返回 tensor 的 GPU 指针，供外部 kernel 零拷贝读写
    void* getDeviceBuffer(const std::string& name) const;

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
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                                stream_;

    // tensor 名 → GPU buffer 指针
    std::unordered_map<std::string, void*>      buffers_;

    void allocateBuffers();
};
