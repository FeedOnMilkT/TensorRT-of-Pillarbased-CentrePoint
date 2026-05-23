#!/usr/bin/env python3
"""
通过 OpenPCDet 的 NuScenesDataset.evaluation() 评测 mini_val 检测结果。

需要外部 OpenPCDet 仓库提供 pcdet 包；通过 --openpcdet-root 或 PYTHONPATH 注入。

用法示例：
  python scripts/eval_mini.py \
      --openpcdet-root ~/OpenPCDet \
      --cfg ~/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes_mini.yaml \
      --data-root /data/sidney/datasets/nuscenes \
      --results-json results/raw_dets.json \
      --output-dir   results/eval_output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet-root", type=Path, default=None,
                    help="OpenPCDet 仓库根目录；不传则依赖 PYTHONPATH 已包含 pcdet。")
    ap.add_argument("--cfg", type=Path, required=True,
                    help="OpenPCDet 配置 yaml（如 cfgs/nuscenes_models/centerpoint_pillar_nuscenes_mini.yaml）")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="nuScenes 根目录（含 samples/、sweeps/、v1.0-* 元数据）")
    ap.add_argument("--results-json", type=Path, required=True,
                    help="C++ pipeline 输出的 raw_dets.json")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="OpenPCDet evaluation() 中间产物目录")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("eval_mini")

    if args.openpcdet_root:
        sys.path.insert(0, str(args.openpcdet_root))

    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.config import cfg, cfg_from_yaml_file

    cfg_from_yaml_file(str(args.cfg), cfg)

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=args.data_root,
        training=False,
        logger=logger,
    )

    print(f"评测帧数: {len(dataset)}")

    with open(args.results_json) as f:
        raw = json.load(f)

    # raw_dets.json key 可能是容器内绝对路径，info['lidar_path'] 是相对路径 (samples/...)。
    # 提取 samples/ 后缀对齐。
    raw_by_rel = {}
    for k, v in raw.items():
        idx = k.find('samples/')
        if idx != -1:
            raw_by_rel[k[idx:]] = v

    name_to_label = {n: i + 1 for i, n in enumerate(cfg.CLASS_NAMES)}

    det_annos = []
    for i in range(len(dataset)):
        info = dataset.infos[i]
        dets = raw_by_rel.get(info['lidar_path'], [])

        if dets:
            names  = np.array([d['class'] for d in dets])
            scores = np.array([d['score'] for d in dets], dtype=np.float32)
            boxes  = np.array([[d['x'], d['y'], d['z'],
                                d['w'], d['l'], d['h'], d['rot'],
                                d.get('vx', 0.0), d.get('vy', 0.0)]
                               for d in dets], dtype=np.float32)
            labels = np.array([name_to_label.get(d['class'], 0) for d in dets], dtype=np.int32)
        else:
            names  = np.array([], dtype=str)
            scores = np.array([], dtype=np.float32)
            boxes  = np.zeros((0, 9), dtype=np.float32)
            labels = np.array([], dtype=np.int32)

        det_annos.append({
            'name':        names,
            'score':       scores,
            'boxes_lidar': boxes,
            'pred_labels': labels,
            'metadata':    {'token': info['token']},
        })

    matched = sum(1 for a in det_annos if len(a['name']) > 0)
    print(f"匹配到预测的帧数: {matched}/{len(det_annos)}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_str, _ = dataset.evaluation(
        det_annos=det_annos,
        class_names=cfg.CLASS_NAMES,
        eval_metric='nuscenes',
        output_path=str(args.output_dir / "trt_result"),
    )

    print(result_str)


if __name__ == "__main__":
    main()
