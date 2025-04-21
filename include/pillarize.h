#pragma once
#include <vector>

struct PillarBatch {
    std::vector<float> features;  // [P, max_pts, 11] flattened
    std::vector<float> mask;      // [P, max_pts]  1=有效点, 0=填充
    std::vector<int>   coords;    // [P, 4]  (batch=0, z=0, y, x)
    int P;
};

PillarBatch pillarize(
    const float* points,   // [N, 5]: x, y, z, intensity, time
    int N,
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    float vx, float vy, float vz,
    int max_pts,
    int max_pillars);
