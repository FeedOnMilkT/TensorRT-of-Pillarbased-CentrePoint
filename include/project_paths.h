#pragma once
// 项目路径解析（header-only）：把硬编码的 /workspace/onnx、/workspace/engines 等
// 全部替换成环境变量驱动的查找。
//   CENTERPOINT_ROOT        项目根（含 CMakeLists.txt 的目录）
//   CENTERPOINT_ONNX_DIR    ONNX 目录，默认 $CENTERPOINT_ROOT/onnx
//   CENTERPOINT_ENGINE_DIR  Engine 目录，默认 $CENTERPOINT_ROOT/engines
//
// 未设 CENTERPOINT_ROOT 时退化为可执行文件所在目录上溯查 CMakeLists.txt；
// 都失败则用当前工作目录，由此可在容器外的任意位置布署。

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>

namespace centerpoint_trt {

inline std::filesystem::path getProjectRoot() {
    if (const char* env = std::getenv("CENTERPOINT_ROOT"); env && *env) {
        return std::filesystem::path(env);
    }
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::error_code ec;
        auto p = std::filesystem::path(buf).parent_path();
        for (; !p.empty() && p != p.root_path(); p = p.parent_path()) {
            if (std::filesystem::exists(p / "CMakeLists.txt", ec)) return p;
        }
        return std::filesystem::path(buf).parent_path();
    }
    return std::filesystem::current_path();
}

inline std::filesystem::path getOnnxDir() {
    if (const char* env = std::getenv("CENTERPOINT_ONNX_DIR"); env && *env) {
        return std::filesystem::path(env);
    }
    return getProjectRoot() / "onnx";
}

inline std::filesystem::path getEngineDir() {
    if (const char* env = std::getenv("CENTERPOINT_ENGINE_DIR"); env && *env) {
        return std::filesystem::path(env);
    }
    return getProjectRoot() / "engines";
}

inline std::string onnxPath(const std::string& name) {
    return (getOnnxDir() / name).string();
}

inline std::string enginePath(const std::string& name) {
    return (getEngineDir() / name).string();
}

}  // namespace centerpoint_trt
