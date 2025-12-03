#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// 一个网络输入 = 一个 CalibInput。
//   binding_name：必须与 ONNX 输入张量名一致（TRT 用名字查找 binding）
//   file_suffix： 在 calib_dir 下找 frame_NNN_<file_suffix>.bin 作为该 input 的数据
//   bytes：       OPT 形状下单帧字节数（与 dump 时 pad 到 OPT_P 的逻辑配套）
struct CalibInput {
    std::string binding_name;
    std::string file_suffix;
    size_t      bytes;
};

// PTQ Entropy-2 校准器。
//
// 在 buildEngine 时由 BuilderConfig::setInt8Calibrator() 持有，TRT 会反复调
// getBatch 喂数据直到返回 false；过程中收集 per-tensor 激活分布并跑 KL 散度
// 选量化阈值。第一次跑完会把分布序列化到 cache_path（writeCalibrationCache），
// 下次构建（同 ONNX + 同硬件）直接命中 cache，不再调 getBatch。
//
// 数据来源：infer_pipeline --dump-calib-dir 产生的 frame_NNN_<suffix>.bin
// 文件，每帧已统一 pad/截断到 OPT 形状，正好对齐 OPT profile 的校准 shape。
class CenterPointCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    CenterPointCalibrator(std::vector<CalibInput> inputs,
                          std::string calib_dir,
                          int num_frames,
                          std::string cache_path);
    ~CenterPointCalibrator() override;

    int  getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void        writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    std::vector<CalibInput>       inputs_;
    std::string                   calib_dir_;
    int                           num_frames_;
    std::string                   cache_path_;
    int                           cur_frame_ = 0;
    std::vector<void*>            d_buffers_;
    std::vector<std::vector<char>> h_scratch_;
    std::vector<char>             cache_;
};