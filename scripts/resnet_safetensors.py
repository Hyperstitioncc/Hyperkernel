"""A torchvision ResNet's checkpoint for hk: DIR/model.safetensors of its weights with each BatchNorm folded into its conv
(eval: w' = w gamma / sqrt(var + eps) per output channel, b' = beta - mean gamma / sqrt(var + eps), in f64, stored F32),
named by the conv: `{conv}.weight` `[o, i, kh, kw]` and `{conv}.bias` `[o]` (conv1, layerL.B.convK, layerL.B.downsample.0),
fc's as stored. `--manifest FILE` also writes its index (manifests/<model>.tsv: name, dtype, shape, count, bytes). No
inference.

    python -I resnet_safetensors.py resnet50|resnet18 OUT_DIR [--manifest FILE]

Weights: torchvision 0.25.0 `ResNet50_Weights.IMAGENET1K_V2` (resnet50-11ad3fa6.pth), `ResNet18_Weights.IMAGENET1K_V1`
(resnet18-f37072fd.pth). Another model's backbone (BEVFormer's, ACT's) folds its own state dict by `folded(state, prefix)`.
"""

import os
import sys

import torch
from safetensors.torch import save_file

WEIGHTS = {"resnet50": "ResNet50_Weights.IMAGENET1K_V2", "resnet18": "ResNet18_Weights.IMAGENET1K_V1"}


def folded(state, prefix="", eps=1e-5):
    """`state`'s convs (names under `prefix`) each with its BatchNorm folded in (conv1+bn1, convK+bnK, downsample.0+.1),
    F32; every other tensor but the norms' as stored (fc's)."""
    out, norms = {}, set()
    for name, w in state.items():
        if not name.startswith(prefix) or not name.endswith(".weight") or w.dim() != 4:
            continue
        conv = name[: -len(".weight")]
        bn = conv[:-1] + "1" if conv.endswith("downsample.0") else conv.replace("conv", "bn")
        g, b, m, v = (state[f"{bn}.{k}"].double() for k in ("weight", "bias", "running_mean", "running_var"))
        scale = g / torch.sqrt(v + eps)
        out[f"{conv}.weight"] = (w.double() * scale[:, None, None, None]).float().contiguous()
        out[f"{conv}.bias"] = (b - m * scale).float().contiguous()
        norms |= {f"{bn}.{k}" for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")}
    for name, t in state.items():
        if name.startswith(prefix) and name not in norms and name not in out and not (name.endswith(".weight") and t.dim() == 4):
            out[name] = t.float().contiguous()
    return out


def model(name):
    """The torchvision model `name` at its pinned weights, eval."""
    import torchvision

    weights = eval(f"torchvision.models.{WEIGHTS[name]}")
    return getattr(torchvision.models, name)(weights=weights).eval(), weights


def manifest(tensors, name, path):
    with open(path, "w") as f:
        f.write(f"# torchvision {WEIGHTS[name]}: its BatchNorms folded into its convs as models/resnet/resnet_safetensors.py writes them (name, dtype, shape, count, bytes)\n")
        for k, t in tensors.items():
            f.write(f"{k}\tF32\t{'x'.join(str(d) for d in t.shape)}\t1\t{t.numel() * 4}\n")


def main(name, out, manifest_path=None):
    m, _ = model(name)
    tensors = folded(m.state_dict())
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    if manifest_path:
        manifest(tensors, name, manifest_path)
    print(f"{len(tensors)} tensors -> {out}/model.safetensors")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[3] if len(a) > 3 and a[2] == "--manifest" else None)
