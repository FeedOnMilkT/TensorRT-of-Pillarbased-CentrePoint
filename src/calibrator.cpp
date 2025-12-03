#include "calibrator.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

CenterPointCalibrator::CenterPointCalibrator(std::vector<CalibInput> inputs,
                                             std::string calib_dir,
                                             int num_frames,
                                             std::string cache_path)
    : inputs_(std::move(inputs)),
      calib_dir_(std::move(calib_dir)),
      num_frames_(num_frames),
      cache_path_(std::move(cache_path))
{
    d_buffers_.resize(inputs_.size(), nullptr);
    h_scratch_.resize(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); i++) {
        cudaMalloc(&d_buffers_[i], inputs_[i].bytes);
        h_scratch_[i].resize(inputs_[i].bytes);
    }
}

CenterPointCalibrator::~CenterPointCalibrator() {
    for (auto* p : d_buffers_) if (p) cudaFree(p);
}

bool CenterPointCalibrator::getBatch(void* bindings[],
                                     const char* names[],
                                     int nbBindings) noexcept {
    if (cur_frame_ >= num_frames_) return false;

    for (int i = 0; i < nbBindings; i++) {
        // 用名字匹配，TRT 不保证 names[] 顺序与构造时 inputs_ 一致
        int idx = -1;
        for (int j = 0; j < (int)inputs_.size(); j++) {
            if (inputs_[j].binding_name == names[i]) { idx = j; break; }
        }
        if (idx < 0) {
            std::cerr << "[calibrator] 未知 binding: " << names[i] << "\n";
            return false;
        }

        char fn[512];
        std::snprintf(fn, sizeof(fn), "%s/frame_%03d_%s.bin",
                      calib_dir_.c_str(), cur_frame_, inputs_[idx].file_suffix.c_str());
        std::ifstream f(fn, std::ios::binary);
        if (!f) {
            std::cerr << "[calibrator] 无法打开: " << fn << "\n";
            return false;
        }
        f.read(h_scratch_[idx].data(), inputs_[idx].bytes);
        if ((size_t)f.gcount() != inputs_[idx].bytes) {
            std::cerr << "[calibrator] 文件大小不符: " << fn
                      << " (got " << f.gcount() << " expected " << inputs_[idx].bytes << ")\n";
            return false;
        }
        cudaMemcpy(d_buffers_[idx], h_scratch_[idx].data(),
                   inputs_[idx].bytes, cudaMemcpyHostToDevice);
        bindings[i] = d_buffers_[idx];
    }

    cur_frame_++;
    if (cur_frame_ % 4 == 0 || cur_frame_ == num_frames_) {
        std::cerr << "[calibrator] " << cur_frame_ << "/" << num_frames_ << " frames\n";
    }
    return true;
}

const void* CenterPointCalibrator::readCalibrationCache(size_t& length) noexcept {
    cache_.clear();
    std::ifstream f(cache_path_, std::ios::binary | std::ios::ate);
    if (!f) { length = 0; return nullptr; }
    size_t bytes = f.tellg();
    f.seekg(0);
    cache_.resize(bytes);
    f.read(cache_.data(), bytes);
    length = cache_.size();
    std::cerr << "[calibrator] 命中 cache: " << cache_path_ << " (" << bytes << " bytes)\n";
    return cache_.data();
}

void CenterPointCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream f(cache_path_, std::ios::binary);
    if (!f) {
        std::cerr << "[calibrator] 无法写入 cache: " << cache_path_ << "\n";
        return;
    }
    f.write(static_cast<const char*>(cache), length);
    std::cerr << "[calibrator] 写入 cache: " << cache_path_ << " (" << length << " bytes)\n";
}