"""facebook/esmfold_v1's pytorch_model.bin (revision 75a3841ee059df2bf4d56688166c8fb459ddd97a) as hk's checkpoint:
DIR/model.safetensors of every floating tensor manifests/esmfold.tsv names, F32 (the ESM-2 tensors stored F16, widened
exactly; the folding head's F32 as stored: the bodies round each to BF16 at its bind), and DIR/config.json. The integer
buffers (af2_to_esm, position_ids) and ESM-2's per-layer rotary inv_freq are not kept: the model states them. No
inference.

    python -I esmfold_safetensors.py pytorch_model.bin config.json OUT_DIR
    python -I esmfold_safetensors.py --manifest pytorch_model.bin > manifests/esmfold.tsv
"""

import os
import shutil
import sys

import torch
from safetensors.torch import save_file

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", "esmfold.tsv")
HEADER = "# facebook/esmfold_v1 75a3841ee059 pytorch_model.bin as esmfold_safetensors.py writes it: each floating tensor F32 (name, dtype, shape, count, bytes)"


def kept(state):
    """Its floating tensors past the rotary tables, F32, in stored order."""
    for name, t in state.items():
        if not t.is_floating_point() or name.endswith("rotary_embeddings.inv_freq"):
            continue
        yield name, t.to(torch.float32).contiguous()


def names():
    with open(MANIFEST) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]
    return [(r[0], tuple(int(x) for x in r[2].split("x"))) for r in rows]


def main(argv):
    if argv[0] == "--manifest":
        state = torch.load(argv[1], map_location="cpu", mmap=True, weights_only=True)
        print(HEADER)
        for name, t in kept(state):
            print(f"{name}\tF32\t{'x'.join(str(d) for d in t.shape)}\t1\t{t.numel() * 4}")
        return
    pth, config, out = argv
    state = dict(kept(torch.load(pth, map_location="cpu", mmap=True, weights_only=True)))
    tensors = {}
    for name, shape in names():
        t = state[name]
        assert tuple(t.shape) == shape, name
        tensors[name] = t
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    shutil.copy(config, os.path.join(out, "config.json"))
    print(f"{len(tensors)} tensors -> {out}/model.safetensors")


if __name__ == "__main__":
    main(sys.argv[1:])
