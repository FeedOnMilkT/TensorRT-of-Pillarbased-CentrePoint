"""
模型结构插桩：打印每个关键模块的输入输出 tensor shape。
在宿主机 openpcdet conda 环境下运行，不进容器。

用法：
  cd /home/uceeanz/OpenPCDet/tools
  python ../../TensorRT/scripts/inspect_shapes.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OpenPCDet'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OpenPCDet' / 'tools'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ── 工具函数 ──────────────────────────────────────────────
def shape_str(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{list(x.shape)} dtype={x.dtype}"
    if isinstance(x, np.ndarray):
        return f"ndarray{list(x.shape)} dtype={x.dtype}"
    return str(type(x))


def print_batch_dict_shapes(tag, batch_dict, keys_of_interest):
    print(f"\n{'='*60}")
    print(f"[{tag}]")
    for k in keys_of_interest:
        if k in batch_dict:
            print(f"  {k}: {shape_str(batch_dict[k])}")
        else:
            print(f"  {k}: (not present)")


# ── 插桩 Hook ─────────────────────────────────────────────
hooks = []

def make_hook(name, keys_before=None, keys_after=None):
    def hook_fn(module, input, output):
        bd = output if isinstance(output, dict) else (input[0] if input else {})
        print(f"\n{'─'*60}")
        print(f"模块: {name}")
        for k in (keys_after or []):
            if k in bd:
                v = bd[k]
                print(f"  OUT {k}: {shape_str(v)}")
    return hook_fn


def register_hooks(model):
    hooks.append(model.vfe.register_forward_hook(
        make_hook("VFE (DynPillarVFE)",
                  keys_after=['pillar_features', 'voxel_coords'])))

    hooks.append(model.map_to_bev_module.register_forward_hook(
        make_hook("MAP_TO_BEV (PointPillarScatter)",
                  keys_after=['spatial_features'])))

    hooks.append(model.backbone_2d.register_forward_hook(
        make_hook("BACKBONE_2D (BaseBEVBackbone)",
                  keys_after=['spatial_features_2d'])))

    hooks.append(model.dense_head.register_forward_hook(
        make_hook("DENSE_HEAD (CenterHead)",
                  keys_after=['pred_dicts'])))


def remove_hooks():
    for h in hooks:
        h.remove()


# ── 主流程 ────────────────────────────────────────────────
def main():
    cfg_file = '/home/uceeanz/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml'
    ckpt    = '/home/uceeanz/OpenPCDet/ckpts/centerpoint_pillar_nuscenes.pth'
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
    print(f"\n数据集大小: {len(dataset)} 帧")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    register_hooks(model)

    data = dataset[0]
    batch = dataset.collate_batch([data])
    load_data_to_gpu(batch)

    # 打印原始输入
    print_batch_dict_shapes("原始输入 (Voxelization 后)", batch,
        ['points', 'voxels', 'voxel_num_points', 'voxel_coords'])

    with torch.no_grad():
        pred_dicts, _ = model.forward(batch)

    remove_hooks()

    # 打印检测头输出（先打印原始结构，再格式化）
    print(f"\n{'='*60}")
    print("[CenterHead 输出 pred_dicts 原始结构]")
    print(f"  type(pred_dicts): {type(pred_dicts)}")
    print(f"  len(pred_dicts): {len(pred_dicts)}")
    print(f"  type(pred_dicts[0]): {type(pred_dicts[0])}")
    if isinstance(pred_dicts[0], (list, tuple)):
        print(f"  len(pred_dicts[0]): {len(pred_dicts[0])}")
        print(f"  type(pred_dicts[0][0]): {type(pred_dicts[0][0])}")
        elem = pred_dicts[0][0]
        if isinstance(elem, dict):
            for k, v in elem.items():
                print(f"    {k}: {shape_str(v)}")
        else:
            print(f"    value: {elem}")
    elif isinstance(pred_dicts[0], dict):
        for k, v in pred_dicts[0].items():
            print(f"  {k}: {shape_str(v)}")

    print(f"\n{'='*60}")
    print("插桩完成。")


if __name__ == '__main__':
    main()
