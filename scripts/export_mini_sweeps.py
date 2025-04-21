#!/usr/bin/env python3
"""
Export nuScenes mini_val point clouds after OpenPCDet sweep aggregation.

Run from OpenPCDet/tools:
  python ../../TensorRT/scripts/export_mini_sweeps.py

Then run inside the TensorRT container:
  build/infer_pipeline --file-list results/mini_sweeps/files.txt --json results/raw_dets.json
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "OpenPCDet"))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OpenPCDet preprocessed mini_val sweeps for TRT inference."
    )
    parser.add_argument(
        "--cfg",
        default=str(REPO_ROOT / "OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes_mini.yaml"),
        help="OpenPCDet model config.",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "OpenPCDet/data/nuscenes"),
        help="nuScenes data root without the version suffix.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "TensorRT/results/mini_sweeps"),
        help="Output directory. The script creates samples/LIDAR_TOP below it.",
    )
    parser.add_argument(
        "--list-path",
        default=None,
        help="Path for the generated file list. Defaults to OUT_DIR/files.txt.",
    )
    parser.add_argument(
        "--list-prefix",
        default=None,
        help="Optional path prefix written to files.txt, useful when OUT_DIR is mounted elsewhere in Singularity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for OpenPCDet's sweep sampling order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("export_mini_sweeps")

    cfg_from_yaml_file(args.cfg, cfg)

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.data_root),
        training=False,
        logger=logger,
    )

    out_dir = Path(args.out_dir)
    list_path = Path(args.list_path) if args.list_path else out_dir / "files.txt"
    file_list = []

    for index, info in enumerate(dataset.infos):
        np.random.seed(args.seed + index)
        points = dataset.get_lidar_with_sweeps(index, max_sweeps=cfg.DATA_CONFIG.MAX_SWEEPS)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 5:
            raise RuntimeError(f"Expected Nx5 points for index {index}, got shape {points.shape}")

        rel_path = Path(info["lidar_path"])
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        points.tofile(out_path)

        if args.list_prefix:
            listed_path = Path(args.list_prefix) / rel_path
        else:
            listed_path = out_path
        file_list.append(str(listed_path))

        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            logger.info("exported %d/%d", index + 1, len(dataset))

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(file_list) + "\n")
    logger.info("wrote file list: %s", list_path)


if __name__ == "__main__":
    main()
