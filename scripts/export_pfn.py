"""
将 DynPillarVFE 的 MLP 部分导出为 ONNX。

DynPillarVFE 内部有 scatter_max（不可导出），这里用 padding + max-pool 等价替换：
  - 输入：padded pillar 点特征 [P, max_pts, C_in]，mask [P, max_pts]
  - 输出：pillar 特征 [P, C_out=64]
权重与预训练 checkpoint 完全兼容，仅聚合方式从 scatter 换成 masked max-pool。

用法（在能 import pcdet 的 Python 环境内）：
  python scripts/export_pfn.py \
      --openpcdet-root ~/OpenPCDet \
      --cfg  ~/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml \
      --ckpt ~/OpenPCDet/ckpts/centerpoint_pillar_nuscenes.pth \
      --data-root /data/sidney/datasets/nuscenes \
      --out $CENTERPOINT_ROOT/onnx/pfn.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def project_root() -> Path:
    env = os.environ.get("CENTERPOINT_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet-root", type=Path, default=None,
                    help="OpenPCDet 仓库根目录；不传则依赖 PYTHONPATH 已含 pcdet。")
    ap.add_argument("--cfg", type=Path, required=True,
                    help="OpenPCDet cfg yaml (centerpoint_pillar_nuscenes.yaml)")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="预训练权重 .pth")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="nuScenes 根目录（仅用于实例化 NuScenesDataset，构图不依赖数据内容）")
    ap.add_argument("--out", type=Path,
                    default=project_root() / "onnx" / "pfn.onnx",
                    help="输出 ONNX 路径，默认 $CENTERPOINT_ROOT/onnx/pfn.onnx")
    ap.add_argument("--data-version", default="v1.0-mini",
                    help="cfg.DATA_CONFIG.VERSION 覆盖值")
    return ap.parse_args()


class PFNExportWrapper(nn.Module):
    """
    将 DynPillarVFE 的两层 PFNLayerV2 重新包装为可导出形式。

    原始 PFNLayerV2 做：linear → BN → ReLU → scatter_max → (cat)
    这里改为：  linear → BN → ReLU → masked_max_pool → (cat)

    masked_max_pool：对 padding 位置填 -inf，再对 max_pts 维度取 max，
    等价于 scatter_max 在无 padding 时的结果。
    """
    def __init__(self, pfn_layers):
        super().__init__()
        self.pfn_layers = pfn_layers

    def forward(self, pillar_features, mask):
        """
        Args:
            pillar_features: [P, max_pts, C_in]
            mask:            [P, max_pts]  1=有效点, 0=padding
        Returns:
            [P, C_out]
        """
        x = pillar_features  # [P, N, C]

        for i, pfn in enumerate(self.pfn_layers):
            P, N, C = x.shape
            x_flat = x.reshape(P * N, C)
            x_flat = pfn.linear(x_flat)
            x_flat = pfn.norm(x_flat) if pfn.use_norm else x_flat
            x_flat = pfn.relu(x_flat)
            x = x_flat.reshape(P, N, -1)        # [P, N, C_out]

            # masked max-pool：padding 位置置 -inf，不参与 max
            inf_mask = (1.0 - mask.unsqueeze(-1).float()) * -1e4
            x_pooled = (x + inf_mask).max(dim=1).values  # [P, C_out]

            if not pfn.last_vfe:
                # cat per-point feat 与 per-pillar feat，送入下一层
                x = torch.cat([x, x_pooled.unsqueeze(1).expand(-1, N, -1)], dim=-1)
            else:
                x = x_pooled

        return x  # [P, 64]


def main():
    args = parse_args()

    if args.openpcdet_root:
        sys.path.insert(0, str(args.openpcdet_root))
        sys.path.insert(0, str(args.openpcdet_root / "tools"))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import NuScenesDataset
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(args.cfg), cfg)
    cfg.DATA_CONFIG.VERSION = args.data_version

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        root_path=args.data_root, training=False, logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
    model.cuda().eval()

    wrapper = PFNExportWrapper(model.vfe.pfn_layers).cuda().eval()

    # P 动态，max_pts=20（config: MAX_POINTS_PER_VOXEL），C_in=11
    P, max_pts, C_in = 12000, 20, 11
    dummy_feats = torch.zeros(P, max_pts, C_in, device='cuda')
    dummy_mask  = torch.ones(P, max_pts, device='cuda')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_feats, dummy_mask),
            str(args.out),
            input_names=['pillar_features', 'pillar_mask'],
            output_names=['pillar_embeddings'],
            dynamic_axes={
                'pillar_features':  {0: 'num_pillars'},
                'pillar_mask':      {0: 'num_pillars'},
                'pillar_embeddings':{0: 'num_pillars'},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    print(f"导出完成：{args.out}")

    import onnx
    m = onnx.load(str(args.out))
    onnx.checker.check_model(m)
    print("ONNX 校验通过")


if __name__ == '__main__':
    main()
