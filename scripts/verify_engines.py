"""
验证 TRT engine 输出与 ONNX (onnxruntime) 数值一致。
在容器内运行：
  cd /workspace
  PYTHONPATH=/workspace/deps python scripts/verify_engines.py
"""

import sys
sys.path.insert(0, '/workspace/deps')

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# ── TRT 工具 ──────────────────────────────────────────────────────────────────

class TRTInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, inputs: dict) -> dict:
        gpu_bufs = {}
        outputs = {}
        stream = cuda.Stream()

        # 输入：set_input_shape + set_tensor_address
        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            gpu_bufs[name] = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(gpu_bufs[name], arr, stream)
            self.context.set_input_shape(name, arr.shape)
            self.context.set_tensor_address(name, int(gpu_bufs[name]))

        # 输出：推断 shape 后分配，set_tensor_address
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                arr = np.empty(shape, dtype=np.float32)
                gpu_bufs[name] = cuda.mem_alloc(arr.nbytes)
                self.context.set_tensor_address(name, int(gpu_bufs[name]))
                outputs[name] = arr

        # 执行
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()

        # 拷回 CPU
        for name, arr in outputs.items():
            cuda.memcpy_dtoh(arr, gpu_bufs[name])

        return outputs


# ── 验证 backbone_head ────────────────────────────────────────────────────────

def verify_backbone():
    print("=" * 60)
    print("验证 backbone_head")

    dummy = np.random.randn(1, 64, 512, 512).astype(np.float32)

    # onnxruntime
    sess = ort.InferenceSession('/workspace/onnx/backbone_head.onnx',
                                providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'spatial_features': dummy})

    # TRT
    trt_infer = TRTInfer('/workspace/engines/backbone_head_fp16.plan')
    trt_out = trt_infer.infer({'spatial_features': dummy})

    # FP32(onnxruntime) vs FP16(TRT)，atol=0.1 是合理上界
    all_pass = True
    ort_names = [o.name for o in sess.get_outputs()]
    for i, name in enumerate(ort_names):
        ort_val = ort_out[i]
        trt_val = trt_out[name]
        close = np.allclose(ort_val, trt_val, atol=0.1, rtol=1e-2)
        max_diff = np.abs(ort_val - trt_val).max()
        status = "PASS" if close else "FAIL"
        print(f"  [{status}] {name:30s} shape={list(ort_val.shape)} max_diff={max_diff:.5f}")
        if not close:
            all_pass = False

    print(f"\nbackbone_head 验证：{'全部通过' if all_pass else '存在差异'}")
    return all_pass


# ── 验证 PFN ─────────────────────────────────────────────────────────────────

def verify_pfn():
    print("=" * 60)
    print("验证 PFN")

    P = 12000
    dummy_feats = np.random.randn(P, 20, 11).astype(np.float32)
    dummy_mask  = np.ones((P, 20), dtype=np.float32)

    # onnxruntime
    sess = ort.InferenceSession('/workspace/onnx/pfn.onnx',
                                providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'pillar_features': dummy_feats,
        'pillar_mask': dummy_mask,
    })[0]

    # TRT
    trt_infer = TRTInfer('/workspace/engines/pfn_fp16.plan')
    trt_out = trt_infer.infer({
        'pillar_features': dummy_feats,
        'pillar_mask': dummy_mask,
    })['pillar_embeddings']

    close = np.allclose(ort_out, trt_out, atol=1e-2, rtol=1e-2)
    max_diff = np.abs(ort_out - trt_out).max()
    print(f"  shape: {list(ort_out.shape)}")
    print(f"  max_diff: {max_diff:.5f}")
    print(f"\nPFN 验证：{'通过' if close else '存在差异'}")
    return close


if __name__ == '__main__':
    ok1 = verify_backbone()
    ok2 = verify_pfn()
    print("\n" + "=" * 60)
    print(f"总体结果：{'全部通过' if ok1 and ok2 else '存在差异，需要排查'}")
