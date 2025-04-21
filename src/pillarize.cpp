#include "pillarize.h"
#include <unordered_map>
#include <algorithm>
#include <cmath>

// 11 个特征的组成（与 DynamicPillarVFE 训练时一致）：
//   [0-4]  x, y, z, intensity, time         原始点特征
//   [5-7]  x-xmean, y-ymean, z-zmean        相对 pillar 内点云重心
//   [8-10] x-xc, y-yc, z-zc                 相对 pillar 体素中心

PillarBatch pillarize(
    const float* points, int N,
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    float vx, float vy, float vz,
    int max_pts, int max_pillars)
{
    const int C = 11;
    const int W = (int)((x_max - x_min) / vx);
    const int H = (int)((y_max - y_min) / vy);

    // 体素中心偏移：第 0 号体素的中心 = min + half_voxel
    const float xc0 = x_min + vx * 0.5f;
    const float yc0 = y_min + vy * 0.5f;
    const float zc0 = z_min + vz * 0.5f;

    // pillar key (py*W+px) → { 分配的 pillar id, 该 pillar 内的点索引列表 }
    struct PillarInfo { int id; std::vector<int> pts; };
    std::unordered_map<int, PillarInfo> pmap;
    pmap.reserve(max_pillars * 2);

    int next_id = 0;

    // 遍历所有点，过滤范围外的点，分配到 pillar
    for (int n = 0; n < N; n++) 
    {
        float x = points[n * 5 + 0];
        float y = points[n * 5 + 1];
        float z = points[n * 5 + 2];

        if (x < x_min || x >= x_max) continue;
        if (y < y_min || y >= y_max) continue;
        if (z < z_min || z >= z_max) continue;

        int px = std::min((int)((x - x_min) / vx), W - 1);
        int py = std::min((int)((y - y_min) / vy), H - 1);
        int key = py * W + px;

        auto it = pmap.find(key);
        if (it == pmap.end()) 
        {
            if (next_id >= max_pillars) continue;
            pmap[key] = { next_id++, {n} };
        } 
        else if ((int)it->second.pts.size() < max_pts) 
        {
            it->second.pts.push_back(n);
        }
    }

    int P = next_id;
    PillarBatch batch;
    batch.P = P;
    batch.features.assign(P * max_pts * C, 0.f);
    batch.mask.assign(P * max_pts, 0.f);
    batch.coords.assign(P * 4, 0);

    for (auto& [key, info] : pmap) {
        int pid = info.id;
        int px  = key % W;
        int py  = key / W;

        batch.coords[pid * 4 + 2] = py;
        batch.coords[pid * 4 + 3] = px;

        int npts = (int)info.pts.size();

        // 计算 pillar 内点云重心（仅用前 npts 个有效点）
        float mx = 0, my = 0, mz = 0;
        for (int i : info.pts) {
            mx += points[i * 5 + 0];
            my += points[i * 5 + 1];
            mz += points[i * 5 + 2];
        }
        mx /= npts; my /= npts; mz /= npts;

        // 当前 pillar 的体素中心
        float xc = px * vx + xc0;
        float yc = py * vy + yc0;
        float zc = 0  * vz + zc0;  // pz=0

        for (int i = 0; i < npts; i++) {
            int ni = info.pts[i];
            float x  = points[ni * 5 + 0];
            float y  = points[ni * 5 + 1];
            float z  = points[ni * 5 + 2];

            float* f = &batch.features[(pid * max_pts + i) * C];
            f[0]  = x;
            f[1]  = y;
            f[2]  = z;
            f[3]  = points[ni * 5 + 3];  // intensity
            f[4]  = points[ni * 5 + 4];  // time
            f[5]  = x - mx;
            f[6]  = y - my;
            f[7]  = z - mz;
            f[8]  = x - xc;
            f[9]  = y - yc;
            f[10] = z - zc;

            batch.mask[pid * max_pts + i] = 1.f;
        }
    }

    return batch;
}
