#!/usr/bin/env python3
"""
nuScenes mAP/NDS 评测脚本（脱离 OpenPCDet）。

读取 C++ pipeline 输出的 raw_dets.json（LiDAR frame 检测框），
做 LiDAR → ego → global 坐标变换，调用官方 nuscenes-devkit 评测。

依赖：nuscenes-devkit、pyquaternion、numpy（直接安装到容器自带 python，参见 CLAUDE.md）。

用法（容器内）：
  python scripts/eval_nuscenes.py \
      --results-json results/raw_dets.json \
      --data-root /data/sidney/datasets/nuscenes \
      --version v1.0-trainval \
      --split val \
      --output-dir results/eval_devkit
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.utils.splits import create_splits_scenes


def pick_attribute(name: str, speed: float) -> str:
    """根据类别和速度推断 attribute（速度阈值 0.2 m/s 与 OpenPCDet 一致）。"""
    if name in ("car", "truck", "bus", "trailer", "construction_vehicle"):
        return "vehicle.moving" if speed > 0.2 else "vehicle.parked"
    if name in ("bicycle", "motorcycle"):
        return "cycle.with_rider" if speed > 0.2 else "cycle.without_rider"
    if name == "pedestrian":
        return "pedestrian.moving" if speed > 0.2 else "pedestrian.standing"
    return ""


def lidar_box_to_global(det, cs_record, ep_record):
    """LiDAR frame 检测框 → global frame。

    输入 det: {x,y,z,w,l,h,rot,vx,vy}
    其中 w=dx（沿 heading），l=dy（垂直 heading），h=dz；rot 为绕 z 轴 yaw。

    返回 (translation[3], rotation Quaternion, velocity_xy[2], size[3]):
      size 顺序按 nuScenes 提交格式 [width, length, height]，
      与 C++ 原始输出相比 w/l 互换。
    """
    q_cs = Quaternion(cs_record["rotation"])
    t_cs = np.asarray(cs_record["translation"], dtype=np.float64)
    q_ep = Quaternion(ep_record["rotation"])
    t_ep = np.asarray(ep_record["translation"], dtype=np.float64)

    # 中心点：lidar → ego → global
    center_lidar = np.array([det["x"], det["y"], det["z"]], dtype=np.float64)
    center_ego = q_cs.rotate(center_lidar) + t_cs
    center_global = q_ep.rotate(center_ego) + t_ep

    # 朝向：以 yaw 构造 lidar frame 四元数，再左乘 cs/ep 旋转
    q_yaw = Quaternion(axis=[0.0, 0.0, 1.0], radians=det["rot"])
    q_global = q_ep * q_cs * q_yaw

    # 速度：方向向量，只旋转不平移
    v_lidar = np.array([det.get("vx", 0.0), det.get("vy", 0.0), 0.0], dtype=np.float64)
    v_global = q_ep.rotate(q_cs.rotate(v_lidar))

    # nuScenes 提交期望 [w, l, h]：w=垂直 heading，l=沿 heading
    size = [float(det["l"]), float(det["w"]), float(det["h"])]

    return center_global, q_global, v_global[:2], size


def build_filename_to_sample(nusc):
    """`samples/LIDAR_TOP/<file>.pcd.bin` → sample_token。"""
    lookup = {}
    for sd in nusc.sample_data:
        if sd["channel"] == "LIDAR_TOP" and sd["is_key_frame"]:
            lookup[sd["filename"]] = sd["sample_token"]
    return lookup


def get_split_sample_tokens(nusc, split: str):
    splits = create_splits_scenes()
    if split not in splits:
        raise ValueError(
            f"未知 split '{split}'，可选: {sorted(splits.keys())}"
        )
    target_scenes = set(splits[split])
    return [
        s["token"]
        for s in nusc.sample
        if nusc.get("scene", s["scene_token"])["name"] in target_scenes
    ]


def normalize_raw_key(key: str) -> str:
    """raw_dets.json key 可能是绝对路径或仓库相对路径，提取 'samples/...' 后缀。"""
    idx = key.find("samples/")
    return key[idx:] if idx != -1 else key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True,
                    help="C++ pipeline 输出的 raw_dets.json")
    ap.add_argument("--data-root", required=True,
                    help="nuScenes 数据根目录（含 samples/、sweeps/、v1.0-* 元数据）")
    ap.add_argument("--version", default="v1.0-mini",
                    help="nuScenes 版本（v1.0-mini / v1.0-trainval / v1.0-test）")
    ap.add_argument("--split", default="mini_val",
                    help="评测 split（mini_val / val / test）")
    ap.add_argument("--output-dir", required=True,
                    help="提交 JSON 与 metrics 输出目录")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # nuscenes-devkit 期望 <dataroot>/<version>/category.json。我们的解压布局可能是
    # /data/sidney/datasets/nuscenes/v1.0-trainval/{samples,sweeps,v1.0-trainval/}，
    # 所以如果传入的 dataroot 下找不到 <version>/category.json 但其子目录 <version>/
    # 下能找到，就自动往下走一层。镜像 export_mini_sweeps.py 的同名修正。
    data_root = Path(args.data_root)
    if not (data_root / args.version / "category.json").exists() and \
       (data_root / args.version / args.version / "category.json").exists():
        data_root = data_root / args.version
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)
    fname_to_token = build_filename_to_sample(nusc)
    split_tokens = set(get_split_sample_tokens(nusc, args.split))
    print(f"split '{args.split}' 含 {len(split_tokens)} 帧")

    with open(args.results_json) as f:
        raw = json.load(f)
    raw_by_fname = {normalize_raw_key(k): v for k, v in raw.items()}
    print(f"raw_dets.json 含 {len(raw_by_fname)} 帧检测")

    results = {tok: [] for tok in split_tokens}
    matched = 0
    skipped_no_token = 0
    skipped_not_in_split = 0

    for fname, dets in raw_by_fname.items():
        sample_token = fname_to_token.get(fname)
        if sample_token is None:
            skipped_no_token += 1
            continue
        if sample_token not in split_tokens:
            skipped_not_in_split += 1
            continue
        matched += 1

        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])

        for det in dets:
            center, q, v_xy, size = lidar_box_to_global(det, cs, ep)
            speed = float(np.linalg.norm(v_xy))
            results[sample_token].append({
                "sample_token": sample_token,
                "translation": center.tolist(),
                "size": size,
                "rotation": [q.w, q.x, q.y, q.z],
                "velocity": v_xy.tolist(),
                "detection_name": det["class"],
                "detection_score": float(det["score"]),
                "attribute_name": pick_attribute(det["class"], speed),
            })

    print(f"匹配进 split 的帧数: {matched}")
    if skipped_no_token:
        print(f"跳过（filename 找不到 sample_token）: {skipped_no_token}")
    if skipped_not_in_split:
        print(f"跳过（sample 不在 {args.split}）: {skipped_not_in_split}")
    empty = sum(1 for v in results.values() if not v)
    if empty:
        print(f"空预测帧（保留空列表，按规范填充）: {empty}/{len(results)}")

    submission = {
        "results": results,
        "meta": {
            "use_camera": False, "use_lidar": True, "use_radar": False,
            "use_map": False, "use_external": False,
        },
    }
    sub_path = out_dir / "results_nusc.json"
    sub_path.write_text(json.dumps(submission))
    print(f"提交文件: {sub_path}")

    cfg = config_factory("detection_cvpr_2019")
    nusc_eval = DetectionEval(
        nusc=nusc,
        config=cfg,
        result_path=str(sub_path),
        eval_set=args.split,
        output_dir=str(out_dir),
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
    print()
    print(f"mean_ap : {metrics_summary['mean_ap']:.4f}")
    print(f"nd_score: {metrics_summary['nd_score']:.4f}")

    # 若 raw_dets.json 同目录存在 bench.json（C++ pipeline --benchmark 产物），顺手打印 FPS
    bench_path = Path(args.results_json).parent / "bench.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        e2e = bench["end_to_end_ms"]
        print()
        print(f"=== Benchmark (from {bench_path.name}, iters={bench['iters']}) ===")
        print(f"FPS (median)   : {bench['fps_median']:.1f}")
        print(f"latency median : {e2e['median']:.2f} ms  "
              f"(p99 {e2e['p99']:.2f},  min {e2e['min']:.2f},  max {e2e['max']:.2f})")
        if bench.get("stages"):
            print("per-stage median (ms):")
            for name, s in bench["stages"].items():
                share = (s["median"] / e2e["median"] * 100.0) if e2e["median"] else 0.0
                print(f"  {name:<14} {s['median']:>8.3f}   ({share:>4.1f}%)")


if __name__ == "__main__":
    main()
