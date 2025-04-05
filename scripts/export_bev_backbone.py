"""
将 BaseBEVBackbone + CenterHead（原始热图输出，无 decode/NMS）导出为 ONNX。

输入：pseudo_image [1, 64, 512, 512]
输出：每个 task head 的原始预测（heatmap, center, center_z, dim, rot, vel）

用法：
  cd /home/uceeanz/OpenPCDet/tools
  python ../../TensorRT/scripts/export_bev_backbone.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OpenPCDet'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'OpenPCDet' / 'tools'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import NuScenesDataset
from pcdet.models import build_network
from pcdet.utils import common_utils


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
    cfg_file  = '/home/uceeanz/OpenPCDet/tools/cfgs/nuscenes_models/centerpoint_pillar_nuscenes.yaml'
    ckpt      = '/home/uceeanz/OpenPCDet/ckpts/centerpoint_pillar_nuscenes.pth'
    out_path  = '/home/uceeanz/TensorRT/onnx/backbone_head.onnx'
    data_root = Path('/home/uceeanz/OpenPCDet/data/nuscenes')

    logger = common_utils.create_logger()
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.DATA_CONFIG.VERSION = 'v1.0-mini'

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        root_path=data_root, training=False, logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
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

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            input_names=['spatial_features'],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )

    print(f"导出完成：{out_path}")
    print(f"输出节点数：{len(output_names)}（{num_tasks} tasks × 6 项）")

    # 快速验证
    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    print("ONNX 校验通过")


if __name__ == '__main__':
    main()
