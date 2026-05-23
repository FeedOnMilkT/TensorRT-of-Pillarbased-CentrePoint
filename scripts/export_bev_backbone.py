"""
将 BaseBEVBackbone + CenterHead（原始热图输出，无 decode/NMS）导出为 ONNX。

输入：pseudo_image [1, 64, 512, 512]
输出：每个 task head 的原始预测（heatmap, center, center_z, dim, rot, vel）

用法（在能 import pcdet 的 Python 环境内）：
  python scripts/export_bev_backbone.py \
      --openpcdet-root ~/OpenPCDet \
      --cfg  ~/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml \
      --ckpt ~/OpenPCDet/ckpts/centerpoint_pillar_nuscenes.pth \
      --data-root /data/sidney/datasets/nuscenes \
      --out $CENTERPOINT_ROOT/onnx/backbone_head.onnx
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
    ap.add_argument("--openpcdet-root", type=Path, default=None)
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=project_root() / "onnx" / "backbone_head.onnx")
    ap.add_argument("--data-version", default="v1.0-mini")
    return ap.parse_args()


class BackboneHeadExportWrapper(nn.Module):
    """
    只导出 backbone_2d + CenterHead 的卷积部分（raw heatmaps）。
    去掉 decode 和 NMS，让 TRT graph 保持纯卷积结构。
    """
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.shared_conv = dense_head.shared_conv
        self.heads_list  = dense_head.heads_list

    def forward(self, spatial_features):
        # backbone
        data_dict = {'spatial_features': spatial_features}
        data_dict = self.backbone_2d(data_dict)
        x = data_dict['spatial_features_2d']

        # shared conv + task heads（raw 输出，不 decode）
        x = self.shared_conv(x)
        outputs = []
        for head in self.heads_list:
            task_out = head(x)
            # 按固定顺序拼成 list，保证 ONNX output 名称稳定
            outputs.append(task_out['hm'])
            outputs.append(task_out['center'])
            outputs.append(task_out['center_z'])
            outputs.append(task_out['dim'])
            outputs.append(task_out['rot'])
            outputs.append(task_out['vel'])
        return tuple(outputs)


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

    wrapper = BackboneHeadExportWrapper(model.backbone_2d, model.dense_head).cuda().eval()

    # dummy input：shape 固定，来自 PointPillarScatter 的输出
    dummy = torch.zeros(1, 64, 512, 512, device='cuda')

    # 构造 output 名称：6 个 task × 6 个预测项
    num_tasks = len(model.dense_head.heads_list)

    output_names = []
    for t in range(num_tasks):
        for name in ['hm', 'center', 'center_z', 'dim', 'rot', 'vel']:
            output_names.append(f'task{t}_{name}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(args.out),
            input_names=['spatial_features'],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )

    print(f"导出完成：{args.out}")
    print(f"输出节点数：{len(output_names)}（{num_tasks} tasks × 6 项）")

    # 快速验证
    import onnx
    m = onnx.load(str(args.out))
    onnx.checker.check_model(m)
    print("ONNX 校验通过")


if __name__ == '__main__':
    main()
