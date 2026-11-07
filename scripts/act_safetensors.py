"""ACT's checkpoint for hk: DIR/model.safetensors of lerobot/act_aloha_sim_transfer_cube_human's tensors with its ResNet-18
backbone's BatchNorms folded into its convs (resnet_safetensors.py `folded`: FrozenBatchNorm2d's eps 1e-5),
every other tensor as stored, and the tick's input-independent values (`hk.*`, F32, computed in f64 by the policy's own
modules (lerobot 0.3.3 modeling_act.py), eval):
  hk.encoder.layers.{j}.self_attn.pos_qkv  [tokens, 3 d]: the encoder's positions (the 1-D table's two rows, the camera
      grid's 2-D sines) times in_proj's q and k rows, zero under v: `(x + pos) Wq = x Wq + pos Wq`, so q | k | v is one
      product over x plus this table;
  hk.decoder.layers.0.norm1.out  [chunk, d]: the decoder's first layer's self-attention block over its zero queries
      (`norm1(0 + self_attn(qpos, qpos, 0))`): input-independent;
  hk.decoder.layers.0.multihead_attn.q  [chunk, d]: cross-attention's q over it (`(n1 + qpos) Wq + bq`);
  hk.decoder.layers.0.multihead_attn.pos_k  [tokens, d]: the encoder's positions times cross-attention's k rows.
config.json copied. `--manifest FILE` also writes its index (manifests/act.tsv: name, dtype, shape, count, bytes). No
inference over any input.

    python -I act_safetensors.py CKDIR OUT_DIR [--manifest FILE]
"""

import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet_safetensors import folded  # noqa: E402

PREFIX = "model.backbone."


def constants(ck):
    """The tick's input-independent values (the module docstring's `hk.*`), by the policy's own modules in f64."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    cfg = PreTrainedConfig.from_pretrained(ck)
    cfg.pretrained_backbone_weights = None
    cfg.device = "cpu"
    m = ACTPolicy.from_pretrained(ck, config=cfg).eval().model.double()
    d = cfg.dim_model
    [(_, image)] = cfg.image_features.items()
    grid = [n // 32 for n in image.shape[1:]]
    out = {}
    with torch.no_grad():
        cam = m.encoder_cam_feat_pos_embed(torch.zeros(1, d, *grid, dtype=torch.float64))
        pos = torch.cat([m.encoder_1d_feature_pos_embed.weight, cam[0].flatten(1).T])
        for j, layer in enumerate(m.encoder.layers):
            w = layer.self_attn.in_proj_weight
            out[f"hk.encoder.layers.{j}.self_attn.pos_qkv"] = torch.cat([pos @ w[: 2 * d].T, torch.zeros(len(pos), d, dtype=pos.dtype)], 1)
        layer = m.decoder.layers[0]
        qpos = m.decoder_pos_embed.weight.unsqueeze(1)
        y = torch.zeros(cfg.chunk_size, 1, d, dtype=torch.float64)
        n1 = layer.norm1(y + layer.self_attn(y + qpos, y + qpos, value=y)[0])[:, 0]
        w, b = layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias
        out["hk.decoder.layers.0.norm1.out"] = n1
        out["hk.decoder.layers.0.multihead_attn.q"] = (n1 + qpos[:, 0]) @ w[:d].T + b[:d]
        out["hk.decoder.layers.0.multihead_attn.pos_k"] = pos @ w[d : 2 * d].T
    return {k: v.float().contiguous() for k, v in out.items()}


def main(ck, out, manifest=None):
    state = load_file(os.path.join(ck, "model.safetensors"))
    tensors = {k: t.contiguous() for k, t in state.items() if not k.startswith(PREFIX)}
    tensors.update(folded(state, PREFIX))
    tensors.update(constants(ck))
    tensors = dict(sorted(tensors.items()))
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    shutil.copy(os.path.join(ck, "config.json"), os.path.join(out, "config.json"))
    if manifest:
        with open(manifest, "w") as f:
            f.write("# lerobot/act_aloha_sim_transfer_cube_human @ ba73b2766f1371cdc133ca4efb97eb090d744625 as models/act/act_safetensors.py writes it: the backbone's BatchNorms folded into its convs, the tick's input-independent values hk.* (name, dtype, shape, count, bytes)\n")
            for k, t in tensors.items():
                f.write(f"{k}\tF32\t{'x'.join(str(d) for d in t.shape)}\t1\t{t.numel() * 4}\n")
    print(f"{len(tensors)} tensors -> {out}/model.safetensors")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[3] if len(a) > 3 and a[2] == "--manifest" else None)
