"""
统计多帧推理中 pillar 数量 P 的分布，用于确定 TRT Optimization Profile 的 min/opt/max。

用法（在能 import pcdet 的 Python 环境内）：
  python scripts/profile_pillar_count.py \
      --openpcdet-root ~/OpenPCDet \
      --cfg  ~/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml \
      --ckpt ~/OpenPCDet/ckpts/centerpoint_pillar_nuscenes.pth \
      --data-root /data/sidney/datasets/nuscenes \
      --frames 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet-root", type=Path, default=None)
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--data-version", default="v1.0-mini")
    ap.add_argument('--frames', type=int, default=20, help='统计帧数')
    return ap.parse_args()


def main():
    args = parse_args()

    if args.openpcdet_root:
        sys.path.insert(0, str(args.openpcdet_root))
        sys.path.insert(0, str(args.openpcdet_root / "tools"))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import NuScenesDataset
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(args.cfg), cfg)
    cfg.DATA_CONFIG.VERSION = args.data_version

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=args.data_root,
        training=False,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    pillar_counts = []

    def vfe_hook(module, input, output):
        p = output['pillar_features'].shape[0]
        pillar_counts.append(p)

    hook = model.vfe.register_forward_hook(vfe_hook)

    n = min(args.frames, len(dataset))
    print(f"\n统计 {n} 帧的 pillar 数量...")
    with torch.no_grad():
        for i in range(n):
            batch = dataset.collate_batch([dataset[i]])
            load_data_to_gpu(batch)
            model.forward(batch)
            print(f"  帧 {i:3d}: P = {pillar_counts[-1]}")

    hook.remove()

    counts = np.array(pillar_counts)
    print(f"\n{'='*40}")
    print(f"统计结果（{n} 帧）")
    print(f"  min : {counts.min()}")
    print(f"  max : {counts.max()}")
    print(f"  mean: {counts.mean():.0f}")
    print(f"  p50 : {np.percentile(counts, 50):.0f}")
    print(f"  p95 : {np.percentile(counts, 95):.0f}")
    print(f"\nTRT Optimization Profile 建议：")
    print(f"  minShapes: pillar_features:{counts.min()}x64x1")
    print(f"  optShapes: pillar_features:{int(np.percentile(counts, 50))}x64x1")
    print(f"  maxShapes: pillar_features:{counts.max()}x64x1")


if __name__ == '__main__':
    main()
