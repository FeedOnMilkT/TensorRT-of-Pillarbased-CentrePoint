#!/usr/bin/env python3
"""
nuScenes mini_val 评测脚本。

用法：
  cd /home/uceeanz/OpenPCDet/tools
  python ../../TensorRT/scripts/eval_mini.py
"""

import json
import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "OpenPCDet"))

from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file
import numpy as np

RAW_DET_JSON = "../../TensorRT/results/raw_dets.json"
CFG_FILE     = "cfgs/nuscenes_models/centerpoint_pillar_nuscenes_mini.yaml"
DATA_ROOT    = Path("../data/nuscenes")
OUTPUT_DIR   = "../../TensorRT/results/eval_output"

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("eval_mini")

    cfg_from_yaml_file(CFG_FILE, cfg)

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=DATA_ROOT,
        training=False,
        logger=logger,
    )

    print(f"mini_val 帧数: {len(dataset)}")

    with open(RAW_DET_JSON) as f:
        raw = json.load(f)

    # raw_dets.json key 是容器绝对路径（/data/nuscenes/v1.0-mini/samples/...）
    # info['lidar_path'] 是相对路径（samples/...），以 samples/ 后缀对齐
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_str, result_dict = dataset.evaluation(
        det_annos=det_annos,
        class_names=cfg.CLASS_NAMES,
        eval_metric='nuscenes',
        output_path=os.path.join(OUTPUT_DIR, "trt_result"),
    )

    print(result_str)


if __name__ == "__main__":
    main()
