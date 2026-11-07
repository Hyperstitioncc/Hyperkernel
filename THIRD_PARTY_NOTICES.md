# Licenses and third-party notices

## This repository
- The repository is under the MIT License (`LICENSE`).
- No third-party source tree is vendored. The build's Rust dependencies come from crates.io under their own licenses
  (`Cargo.lock` lists them).

## The kernel bundles
`kernels/b300/*` hold binaries (cubins, their SASS listings, and the data their load writes) that HyperKernels compiled
from its own sources.

## Interoperability
- The server serves the DeepSeek-V4.1-Flash checkpoint (DeepSeek's license) and its tokenizer; neither is included here.
  A bundle's engram token map is derived from that tokenizer.
- `drive --model openpilot` runs comma.ai's openpilot driving model (openpilot v0.11.2, MIT License, Copyright (c) 2018
  Comma.ai, Inc.); `dna --model evo2-7b` runs Arc Institute's Evo 2 7B (`arcinstitute/evo2_7b`, Apache License 2.0);
  `world --model mg3` runs Skywork's Matrix-Game 3.0 (`Skywork/Matrix-Game-3.0`, Apache License 2.0). None of their
  weights is included here.
- `robot --model act` runs LeRobot's ACT policy (`lerobot/act_aloha_sim_transfer_cube_human`, Apache License 2.0);
  `scripts/act_safetensors.py` reads it with LeRobot's own modules. `fold --model esmfold` runs Meta's ESMFold
  (`facebook/esmfold_v1`, MIT License). Neither's weights is included here.
- `robot --model pi0-fast` runs Physical Intelligence's π₀-FAST as LeRobot publishes it (`lerobot/pi0fast-base`), whose
  weights are released under the Gemma Terms of Use (it builds on Google's PaliGemma); they are not included here, and a
  user downloads them under those terms. The prompt's ids are given already tokenized (no tokenizer is included).
- The AgentX benchmark client (`scripts/dsv41/dsv41 bench`) is `agentx-harness` from PyPI, installed by `setup`.
