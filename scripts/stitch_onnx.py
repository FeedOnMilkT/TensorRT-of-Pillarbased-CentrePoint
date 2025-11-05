"""把 PFN 和 backbone_head 两段 ONNX 拼成一个端到端图，中间插入 PillarScatter 自定义节点。

输入图：
  pfn.onnx            inputs : pillar_features [P,20,11], pillar_mask [P,20]
                      output : pillar_embeddings [P, 64]
  backbone_head.onnx  input  : spatial_features [1,64,512,512]
                      outputs: 36 个 head tensors

输出图 centerpoint_e2e.onnx：
  inputs : pillar_features [P,20,11]
           pillar_mask     [P,20]
           pillar_coords   [P, 4] (新增)
  outputs: 36 个 head tensors

用法（容器内）：
  PYTHONPATH=/workspace/deps python3 scripts/stitch_onnx.py
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


# Canvas 几何参数：与训练配置 cbgs_pp_multihead 一致
H = 512
W = 512
C = 64


def stitch(pfn_path: str, bb_path: str, out_path: str) -> None:
    pfn = gs.import_onnx(onnx.load(pfn_path))
    bb = gs.import_onnx(onnx.load(bb_path))

    # ── 1. 校验对接面 ───────────────────────────────────────────────────────
    assert len(pfn.outputs) == 1, f"PFN 期望 1 个输出，得到 {len(pfn.outputs)}"
    pfn_out = pfn.outputs[0]
    assert pfn_out.name == "pillar_embeddings", f"PFN 输出名意外：{pfn_out.name}"

    bb_input = next((t for t in bb.inputs if t.name == "spatial_features"), None)
    assert bb_input is not None, "backbone 没有 spatial_features 输入"

    print(f"[stitch] PFN output : {pfn_out.name} {pfn_out.shape} {pfn_out.dtype}")
    print(f"[stitch] BB  input  : {bb_input.name} {bb_input.shape} {bb_input.dtype}")

    # ── 2. 给 backbone 的内部 tensor / node 加前缀，避免和 PFN 命名冲突 ─────
    #     (initializer / 权重也跟着改，不改 IO 输入输出名)
    bb_io_names = {t.name for t in bb.inputs} | {t.name for t in bb.outputs}
    seen = set()
    for node in bb.nodes:
        if node.name and not node.name.startswith("bb_"):
            node.name = "bb_" + node.name
        for v in list(node.inputs) + list(node.outputs):
            if v.name in bb_io_names or v.name in seen:
                continue
            seen.add(v.name)
            v.name = "bb_" + v.name

    # ── 3. 新增 graph 输入 pillar_coords ────────────────────────────────────
    pillar_coords = gs.Variable(
        name="pillar_coords",
        dtype=np.int32,
        shape=("num_pillars", 4),
    )

    # Scatter 输出 tensor：shape 由 plugin attribute 决定，dtype = float32
    scatter_out = gs.Variable(
        name="canvas",
        dtype=np.float32,
        shape=(1, C, H, W),
    )

    # ── 4. 把 backbone 中所有以 spatial_features 为输入的连接改到 scatter_out ─
    n_replaced = 0
    for node in bb.nodes:
        for i, inp in enumerate(list(node.inputs)):
            if inp is bb_input:
                node.inputs[i] = scatter_out
                n_replaced += 1
    print(f"[stitch] 改写了 {n_replaced} 处 spatial_features 引用")
    assert n_replaced > 0, "spatial_features 没有任何下游消费者，输入名是不是变了？"

    # ── 5. 构造 PillarScatter 自定义算子节点 ────────────────────────────────
    scatter_node = gs.Node(
        op="PillarScatter",
        name="PillarScatter_0",
        domain="centerpoint_trt",            # 与 plugin namespace 一致
        inputs=[pfn_out, pillar_coords],
        outputs=[scatter_out],
        attrs={"H": H, "W": W, "C": C},
    )

    # ── 6. 合并 graph：pfn 节点 + scatter + bb 节点 ──────────────────────────
    merged = gs.Graph(
        nodes=pfn.nodes + [scatter_node] + bb.nodes,
        inputs=list(pfn.inputs) + [pillar_coords],
        outputs=list(bb.outputs),
        opset=max(pfn.opset, bb.opset),
    )
    merged.cleanup().toposort()

    # ── 7. 导出 ─────────────────────────────────────────────────────────────
    model = gs.export_onnx(merged)
    # 每个 domain 只保留一条 opset_import；缺自定义 domain 则补一条
    seen = set()
    deduped = []
    for o in model.opset_import:
        if o.domain in seen:
            continue
        seen.add(o.domain)
        deduped.append(o)
    if "centerpoint_trt" not in seen:
        deduped.append(onnx.helper.make_opsetid("centerpoint_trt", 1))
    del model.opset_import[:]
    model.opset_import.extend(deduped)
    onnx.save(model, out_path)
    print(f"[stitch] 写出 {out_path}")
    print(f"[stitch] 最终 graph 输入：{[t.name for t in merged.inputs]}")
    print(f"[stitch] 最终 graph 输出数：{len(merged.outputs)}")
    print(f"[stitch] 节点总数：{len(merged.nodes)}")


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--pfn", default=str(root / "onnx" / "pfn.onnx"))
    parser.add_argument("--backbone", default=str(root / "onnx" / "backbone_head.onnx"))
    parser.add_argument("--out", default=str(root / "onnx" / "centerpoint_e2e.onnx"))
    args = parser.parse_args()
    stitch(args.pfn, args.backbone, args.out)


if __name__ == "__main__":
    main()
