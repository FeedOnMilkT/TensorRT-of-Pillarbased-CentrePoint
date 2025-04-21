#pragma once
#include <vector>
#include <string>

struct Detection {
    float x, y, z;
    float w, l, h;
    float rot;
    float vx, vy;
    float score;
    int   cls_id;   // 全局类别 id（跨所有 task）
    int   task_id;
};

// 解码一个 task 的 CenterHead 输出，返回得分超过阈值的候选框
// hm       [1, num_cls, H, W]  原始 logit（函数内部做 sigmoid）
// center   [1, 2,       H, W]  x/y 偏移（grid 单位）
// center_z [1, 1,       H, W]  z 绝对高度
// dim      [1, 3,       H, W]  log(dx,dy,dz)
// rot      [1, 2,       H, W]  (cos, sin)
// vel      [1, 2,       H, W]  (vx, vy) m/s
std::vector<Detection> decodeTask(
    const float* hm, const float* center, const float* center_z,
    const float* dim, const float* rot,   const float* vel,
    int H, int W, int num_cls, int task_id, int cls_offset,
    float score_thresh,
    float x_min, float y_min, float vx, float vy, int out_stride);

// Circle NMS：按得分降序贪心抑制，保留圆心距离 > radius 的框
std::vector<Detection> circleNMS(std::vector<Detection> dets, float radius);
