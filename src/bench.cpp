#include "bench.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

StageTimer::StageTimer() {
    cudaEventCreate(&frame_start_);
    cudaEventCreate(&frame_stop_);
}

StageTimer::~StageTimer() {
    cudaEventDestroy(frame_start_);
    cudaEventDestroy(frame_stop_);
}

void StageTimer::begin(const std::string& stage) {
    pending_cpu_[stage] = Clock::now();
}

void StageTimer::end(const std::string& stage) {
    auto it = pending_cpu_.find(stage);
    if (it == pending_cpu_.end()) {
        throw std::runtime_error("StageTimer::end without matching begin: " + stage);
    }
    auto dur = std::chrono::duration<float, std::milli>(Clock::now() - it->second).count();
    pending_cpu_.erase(it);

    if (samples_ms_.find(stage) == samples_ms_.end()) {
        stage_order_.push_back(stage);
    }
    samples_ms_[stage].push_back(dur);
}

void StageTimer::beginFrameGpu(cudaStream_t stream) {
    cudaEventRecord(frame_start_, stream);
    frame_active_ = true;
}

void StageTimer::endFrameGpu(cudaStream_t stream) {
    if (!frame_active_) return;
    cudaEventRecord(frame_stop_, stream);
}

void StageTimer::commitFrame() {
    if (!frame_active_) return;
    cudaEventSynchronize(frame_stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, frame_start_, frame_stop_);
    end_to_end_ms_.push_back(ms);
    frame_active_ = false;
}

void StageTimer::reset() {
    samples_ms_.clear();
    stage_order_.clear();
    end_to_end_ms_.clear();
    pending_cpu_.clear();
    frame_active_ = false;
}

int StageTimer::sampleCount() const {
    return static_cast<int>(end_to_end_ms_.size());
}

namespace {

struct Stats {
    float mean, median, p99, vmin, vmax;
};

Stats compute(std::vector<float> v) {
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.vmin = v.front();
    s.vmax = v.back();
    s.median = v[v.size() / 2];
    size_t p99i = static_cast<size_t>(std::min<float>(
        v.size() - 1, std::floor(v.size() * 0.99f)));
    s.p99 = v[p99i];
    s.mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    return s;
}

}  // namespace

void StageTimer::summary(std::ostream& os) const {
    if (end_to_end_ms_.empty()) {
        os << "StageTimer: no samples\n";
        return;
    }

    Stats e2e = compute(end_to_end_ms_);

    os << "\n=== Benchmark (iters=" << end_to_end_ms_.size()
       << ", end_to_end via cudaEvent, per-stage via chrono) ===\n";
    os << std::left << std::setw(18) << "Stage"
       << std::right
       << std::setw(9) << "mean"
       << std::setw(9) << "median"
       << std::setw(9) << "min"
       << std::setw(9) << "max"
       << std::setw(9) << "p99"
       << std::setw(9) << "share"
       << "\n";
    os << std::string(72, '-') << "\n";
    os << std::fixed << std::setprecision(3);

    for (const auto& name : stage_order_) {
        Stats s = compute(samples_ms_.at(name));
        float share = e2e.median > 0 ? (s.median / e2e.median * 100.0f) : 0.0f;
        os << std::left << std::setw(18) << name
           << std::right
           << std::setw(9) << s.mean
           << std::setw(9) << s.median
           << std::setw(9) << s.vmin
           << std::setw(9) << s.vmax
           << std::setw(9) << s.p99
           << std::setw(8) << std::setprecision(1) << share << "%"
           << std::setprecision(3)
           << "\n";
    }
    os << std::string(72, '-') << "\n";
    os << std::left << std::setw(18) << "end_to_end"
       << std::right
       << std::setw(9) << e2e.mean
       << std::setw(9) << e2e.median
       << std::setw(9) << e2e.vmin
       << std::setw(9) << e2e.vmax
       << std::setw(9) << e2e.p99
       << "\n";

    float fps = e2e.median > 0 ? 1000.0f / e2e.median : 0.0f;
    os << "\nFPS (1000 / median end_to_end): " << std::setprecision(1) << fps
       << " fps  (latency median " << std::setprecision(2) << e2e.median << " ms)\n";
}

void StageTimer::writeJson(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("无法写入 bench JSON: " + path);

    Stats e2e = compute(end_to_end_ms_);
    float fps = e2e.median > 0 ? 1000.0f / e2e.median : 0.0f;

    f << "{\n";
    f << "  \"iters\": " << end_to_end_ms_.size() << ",\n";
    f << std::fixed << std::setprecision(4);
    f << "  \"fps_median\": " << fps << ",\n";
    f << "  \"end_to_end_ms\": {"
      << "\"mean\": " << e2e.mean
      << ", \"median\": " << e2e.median
      << ", \"min\": " << e2e.vmin
      << ", \"max\": " << e2e.vmax
      << ", \"p99\": " << e2e.p99 << "},\n";
    f << "  \"stages\": {\n";
    for (size_t i = 0; i < stage_order_.size(); i++) {
        const auto& name = stage_order_[i];
        Stats s = compute(samples_ms_.at(name));
        f << "    \"" << name << "\": {"
          << "\"mean\": " << s.mean
          << ", \"median\": " << s.median
          << ", \"min\": " << s.vmin
          << ", \"max\": " << s.vmax
          << ", \"p99\": " << s.p99 << "}";
        if (i + 1 < stage_order_.size()) f << ",";
        f << "\n";
    }
    f << "  }\n";
    f << "}\n";
}
