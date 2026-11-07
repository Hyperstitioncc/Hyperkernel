import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

T = "p_policy_model.temporal_summarizer."
MATMULS = {
    "model/p_val_13": (T + "_feats_encode.0.weight", (512, 512)),
    "model/p_val_16": (T + "_feats_encode.2.weight", (512, 512)),
    "model/p_val_29": (T + "transformer.0.attn.c_attn.weight", (512, 1536)),
    "model/p_val_58": (T + "transformer.0.attn.c_proj.weight", (512, 512)),
    "model/p_val_62": (T + "transformer.0.mlp.c_fc.weight", (512, 2048)),
    "model/p_val_64": (T + "transformer.0.mlp.c_proj.weight", (2048, 512)),
}
POS = ("model/p_add", T + "pos")
NOT = {"model/scalar_tensor_default", "model/p_scalar_tensor_default", "model/p_val_51", "model/p_pad"}
NORM = {"model/vision_model.vision._mean": 127.5, "model/vision_model.vision._std": 63.75}
FLOATS = (np.float16, np.float32)


def need(ok, *why):
    if not ok:
        raise SystemExit("openpilot_safetensors: " + " ".join(map(str, why)))


def tensors(path):
    m = onnx.load(path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    by_out = {n.output[0]: n for n in m.graph.node}
    out, named = {}, set()
    for node_out, (name, shape) in MATMULS.items():
        n = by_out.get(node_out)
        need(n is not None and n.op_type == "MatMul" and n.input[1] in inits, node_out, "is not a MatMul of an initializer")
        w = inits[n.input[1]]
        need(w.shape == shape, name, w.shape, "not", shape)
        out[name] = np.ascontiguousarray(w.T.astype(np.float32))
        named.add(n.input[1])
    n = by_out.get(POS[0])
    need(n is not None and n.op_type == "Add" and n.input[1] in inits and inits[n.input[1]].shape == (1, 9, 512), POS[0], "is not an Add of a (1, 9, 512) initializer")
    out[POS[1]] = np.ascontiguousarray(inits[n.input[1]][0].astype(np.float32))
    named.add(n.input[1])
    for k, c in NORM.items():
        need(k in inits and np.all(inits[k] == c), k, "is not", c)
    for k, v in inits.items():
        if k in named or k in NOT or k in NORM or v.dtype not in FLOATS:
            continue
        need(k.startswith("model/"), k, "is outside model/")
        out[k[len("model/"):]] = np.ascontiguousarray(v.astype(np.float32))
    return out


def main(path, dst):
    t = tensors(path)
    os.makedirs(dst, exist_ok=True)
    save_file(t, os.path.join(dst, "model.safetensors"))
    print(f"{len(t)} tensors -> {dst}/model.safetensors")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
