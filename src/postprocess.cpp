#include "postprocess.h"
#include <cmath>
#include <algorithm>

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

std::vector<Detection> decodeTask(
    const float* hm, const float* center, const float* center_z,
    const float* dim, const float* rot, const float* vel,
    int H, int W, int num_cls, int task_id, int cls_offset,
    float score_thresh,
    float x_min, float y_min, float vx, float vy, int out_stride)
{
    // NCHW 访问辅助：tensor[c, gy, gx]
    auto at = [&](const float* t, int c, int gy, int gx) {
        return t[c * H * W + gy * W + gx];
    };

    std::vector<Detection> dets;

    for (int c = 0; c < num_cls; c++) {
        for (int gy = 0; gy < H; gy++) {
            for (int gx = 0; gx < W; gx++) {
                float score = sigmoid(at(hm, c, gy, gx));
                if (score < score_thresh) continue;

                // 局部极大值检验（3×3 邻域）：过滤掉非峰值格子
                bool is_peak = true;
                for (int dy = -1; dy <= 1 && is_peak; dy++) {
                    for (int dx = -1; dx <= 1 && is_peak; dx++) {
                        if (dy == 0 && dx == 0) continue;
                        int ny = gy + dy, nx = gx + dx;
                        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                        if (sigmoid(at(hm, c, ny, nx)) > score) is_peak = false;
                    }
                }
                if (!is_peak) continue;

                Detection d;
                d.score   = score;
                d.task_id = task_id;
                d.cls_id  = cls_offset + c;

                // 反算真实坐标：grid 位置 + 偏移 → 乘以步长和体素尺寸 → 加上原点
                d.x = (gx + at(center, 0, gy, gx)) * out_stride * vx + x_min;
                d.y = (gy + at(center, 1, gy, gx)) * out_stride * vy + y_min;
                d.z = at(center_z, 0, gy, gx);
                d.w = expf(at(dim, 0, gy, gx));
                d.l = expf(at(dim, 1, gy, gx));
                d.h = expf(at(dim, 2, gy, gx));
                d.rot = atan2f(at(rot, 1, gy, gx), at(rot, 0, gy, gx));
                d.vx  = at(vel, 0, gy, gx);
                d.vy  = at(vel, 1, gy, gx);

                dets.push_back(d);
            }
        }
    }
    return dets;
}

std::vector<Detection> circleNMS(std::vector<Detection> dets, float radius) {
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> kept;

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            float dx = dets[i].x - dets[j].x;
            float dy = dets[i].y - dets[j].y;
            if (sqrtf(dx * dx + dy * dy) < radius) suppressed[j] = true;
        }
    }
    return kept;
}
