"""
统计多帧推理中 pillar 数量 P 的分布，用于确定 TRT Optimization Profile 的 min/opt/max。

用法：
  cd /home/uceeanz/OpenPCDet/tools
  python ../../TensorRT/scripts/profile_pillar_count.py --frames 20
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OpenPCDet'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OpenPCDet' / 'tools'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=20, help='统计帧数')
    args = parser.parse_args()

    cfg_file = '/home/uceeanz/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml'
    ckpt     = '/home/uceeanz/OpenPCDet/ckpts/centerpoint_pillar_nuscenes.pth'
    data_root = Path('/home/uceeanz/OpenPCDet/data/nuscenes')

    logger = common_utils.create_logger()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.DATA_CONFIG.VERSION = 'v1.0-mini'

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=data_root,
        training=False,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
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
