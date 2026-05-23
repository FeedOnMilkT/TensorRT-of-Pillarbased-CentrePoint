#!/usr/bin/env python3
"""
Export nuScenes mini_val LiDAR frames (current + 9 sweeps aggregated) for TRT inference.

Standalone: depends only on nuscenes-devkit, pyquaternion, numpy. No OpenPCDet import.

Run (容器内，参数化路径)：
  python scripts/export_mini_sweeps.py \
      --data-root /data/sidney/datasets/nuscenes \
      --version v1.0-mini --split val \
      --out-dir   /data/sidney/datasets/nuscenes_sweeps_cache/mini_val \
      --list-path /data/sidney/datasets/nuscenes_sweeps_cache/mini_val/files.txt

Then inside the TensorRT container:
  build/infer_pipeline --file-list <out-dir>/files.txt --json results/raw_dets.json

The output bin layout matches OpenPCDet's NuScenesDataset.get_lidar_with_sweeps with
MAX_SWEEPS=10: each .bin is float32 [N, 5] where columns are (x, y, z, intensity, time_lag).
"""

import argparse
import logging
from functools import reduce
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


SPLIT_SCENES = {
    "v1.0-mini": ("mini_train", "mini_val"),
    "v1.0-trainval": ("train", "val"),
    "v1.0-test": ("test", None),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export nuScenes mini_val sweep-aggregated LiDAR frames."
    )
    parser.add_argument(
        "--data-root",
        default="/data/sidney/datasets/nuscenes",
        help="nuScenes data root (contains the version subdirectory).",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        choices=list(SPLIT_SCENES),
        help="nuScenes release.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Split to export.",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/sidney/datasets/nuscenes_sweeps_cache/mini_val",
        help="Output directory. Aggregated bins live under OUT_DIR/samples/LIDAR_TOP/.",
    )
    parser.add_argument(
        "--list-path",
        default=None,
        help="Path for the generated file list. Defaults to OUT_DIR/files.txt.",
    )
    parser.add_argument(
        "--list-prefix",
        default=None,
        help="Optional path prefix written to files.txt (useful when OUT_DIR is "
             "mounted elsewhere in the inference container).",
    )
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Per-frame seed offset; matches OpenPCDet export script behaviour.",
    )
    return parser.parse_args()


def remove_ego_points(points: np.ndarray, center_radius: float = 1.0) -> np.ndarray:
    mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]


def build_sweeps(nusc: NuScenes, sample, ref_data_path: Path, max_sweeps: int):
    """Replicates OpenPCDet's fill_trainval_infos sweep list (length max_sweeps - 1)."""
    ref_sd_token = sample["data"]["LIDAR_TOP"]
    ref_sd_rec = nusc.get("sample_data", ref_sd_token)
    ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
    ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
    ref_time = 1e-6 * ref_sd_rec["timestamp"]

    ref_from_car = transform_matrix(
        ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
    )
    car_from_global = transform_matrix(
        ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True
    )

    curr_sd_rec = ref_sd_rec
    sweeps = []
    while len(sweeps) < max_sweeps - 1:
        if curr_sd_rec["prev"] == "":
            if len(sweeps) == 0:
                sweeps.append({
                    "lidar_path": str(ref_data_path),
                    "transform_matrix": None,
                    "time_lag": 0.0,
                })
            else:
                sweeps.append(sweeps[-1])
        else:
            curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])
            cur_pose = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
            cur_cs = nusc.get("calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])

            global_from_car = transform_matrix(
                cur_pose["translation"], Quaternion(cur_pose["rotation"]), inverse=False
            )
            car_from_current = transform_matrix(
                cur_cs["translation"], Quaternion(cur_cs["rotation"]), inverse=False
            )
            tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            sweep_path = Path(nusc.get_sample_data_path(curr_sd_rec["token"]))
            sweeps.append({
                "lidar_path": str(sweep_path),
                "transform_matrix": tm,
                "time_lag": ref_time - 1e-6 * curr_sd_rec["timestamp"],
            })
    return sweeps


def load_sweep(sweep: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.fromfile(sweep["lidar_path"], dtype=np.float32, count=-1).reshape(-1, 5)[:, :4]
    points = remove_ego_points(points).T
    if sweep["transform_matrix"] is not None:
        n = points.shape[1]
        points[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points[:3, :], np.ones(n))))[:3, :]
    times = sweep["time_lag"] * np.ones((1, points.shape[1]))
    return points.T, times.T


def aggregate(nusc: NuScenes, sample, max_sweeps: int) -> np.ndarray:
    ref_path = Path(nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"]))
    points = np.fromfile(str(ref_path), dtype=np.float32, count=-1).reshape(-1, 5)[:, :4]

    sweep_pts = [points]
    sweep_times = [np.zeros((points.shape[0], 1), dtype=points.dtype)]

    sweeps = build_sweeps(nusc, sample, ref_path, max_sweeps)
    for k in np.random.choice(len(sweeps), max_sweeps - 1, replace=False):
        pts, times = load_sweep(sweeps[k])
        sweep_pts.append(pts)
        sweep_times.append(times)

    points = np.concatenate(sweep_pts, axis=0)
    times = np.concatenate(sweep_times, axis=0).astype(points.dtype)
    return np.concatenate((points, times), axis=1).astype(np.float32)


def collect_split_samples(nusc: NuScenes, version: str, split: str):
    train_split, val_split = SPLIT_SCENES[version]
    target = train_split if split == "train" else val_split
    if target is None:
        raise ValueError(f"Split '{split}' is not defined for version '{version}'.")
    target_scene_names = set(getattr(splits, target))

    name_to_token = {scene["name"]: scene["token"] for scene in nusc.scene}
    available = set(name_to_token).intersection(target_scene_names)
    target_tokens = {name_to_token[name] for name in available}

    return [s for s in nusc.sample if s["scene_token"] in target_tokens]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("export_mini_sweeps")

    # nuScenes-devkit expects dataroot to contain `<version>/category.json` plus
    # `samples/` and `sweeps/`. OpenPCDet's layout puts these one level below
    # `--data-root`, so we descend into the version directory here and reuse the
    # same path as the relative-base for output filenames.
    data_root = Path(args.data_root)
    version_root = data_root / args.version
    if not (version_root / args.version / "category.json").exists() and \
       (data_root / args.version / "category.json").exists() and \
       (data_root / "samples").exists():
        # Caller already passed the directory holding samples/sweeps/<version>/.
        version_root = data_root

    nusc = NuScenes(version=args.version, dataroot=str(version_root), verbose=False)
    samples = collect_split_samples(nusc, args.version, args.split)
    logger.info("found %d samples in %s/%s", len(samples), args.version, args.split)

    out_dir = Path(args.out_dir)
    list_path = Path(args.list_path) if args.list_path else out_dir / "files.txt"
    file_list: list[str] = []

    for index, sample in enumerate(samples):
        np.random.seed(args.seed + index)
        points = aggregate(nusc, sample, args.max_sweeps)
        if points.ndim != 2 or points.shape[1] != 5:
            raise RuntimeError(f"Expected Nx5 points for index {index}, got shape {points.shape}")

        ref_abs = Path(nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"]))
        rel_path = ref_abs.relative_to(version_root)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        points.tofile(out_path)

        listed_path = Path(args.list_prefix) / rel_path if args.list_prefix else out_path
        file_list.append(str(listed_path))

        if (index + 1) % 20 == 0 or index + 1 == len(samples):
            logger.info("exported %d/%d", index + 1, len(samples))

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(file_list) + "\n")
    logger.info("wrote file list: %s", list_path)


if __name__ == "__main__":
    main()
