#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

// 简单的分阶段计时器：
//   - per-stage 用 std::chrono 测 wall clock（前提是每个 stage 内部已经
//     cudaStreamSynchronize；本 pipeline 当前满足）
//   - end_to_end 用 cudaEvent 跨 stream 测整帧时间，符合 CLAUDE.md 第 689 行规范
//
// 调用顺序（每帧）：
//   timer.beginFrameGpu(stream);
//   timer.begin("pillarize"); ... timer.end("pillarize");
//   timer.begin("pfn");       ... timer.end("pfn");
//   ...
//   timer.endFrameGpu(stream);
//   timer.commitFrame();
class StageTimer {
public:
    StageTimer();
    ~StageTimer();

    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

    void begin(const std::string& stage);
    void end(const std::string& stage);

    void beginFrameGpu(cudaStream_t stream);
    void endFrameGpu(cudaStream_t stream);

    // 同步本帧所有 GPU event 并把样本归档；CPU stage 直接归档
    void commitFrame();

    // 丢弃已累积样本（通常 warmup 之后调用）
    void reset();

    void summary(std::ostream& os) const;
    void writeJson(const std::string& path) const;

    int sampleCount() const;

private:
    using Clock = std::chrono::high_resolution_clock;

    std::map<std::string, Clock::time_point> pending_cpu_;
    std::map<std::string, std::vector<float>> samples_ms_;
    std::vector<std::string> stage_order_;

    cudaEvent_t frame_start_;
    cudaEvent_t frame_stop_;
    bool frame_active_ = false;
    std::vector<float> end_to_end_ms_;
};
