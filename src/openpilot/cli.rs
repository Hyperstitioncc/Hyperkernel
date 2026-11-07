use std::path::{Path, PathBuf};

use clap::Args;

use crate::bundle::Bundle;
use crate::gpu::{cuda, load, ranks};
use crate::openpilot::drive::{self, FRAME_BYTES};
use crate::openpilot::{Device, Openpilot};
use crate::weights::Checkpoint;

pub const GPUS: usize = 1;

#[derive(Args)]
pub struct Flags {
    #[arg(long, help = "A directory holding new_img.u8: raw u8 camera frames, back to back")]
    frames: PathBuf,
    #[arg(long, help = "Frames to drive [default: all in new_img.u8]")]
    count: Option<usize>,
    #[arg(long, help = "A directory to write outputs.f32 into: the model's output row per frame")]
    out: Option<PathBuf>,
}

pub fn bundle(_: &Flags) -> &'static str {
    "openpilot"
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let path = flags.frames.join("new_img.u8");
    let frames = std::fs::read(&path).map_err(|e| format!("--frames {}: {e}", path.display()))?;
    let n = flags.count.unwrap_or(frames.len() / FRAME_BYTES);
    let bundle = Bundle::<Openpilot>::read(kernels, 0)?;
    cuda::visible(GPUS)?;
    let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric)?;
    let loaded = load::load(&bundle, &ranks, Some(&Checkpoint::open(weights)?), &[])?;
    let d = drive::drive(&frames, n, &mut Device { bundle: &bundle, loaded })?;
    print!("{}", d.lines);
    if let Some(dir) = flags.out {
        std::fs::create_dir_all(&dir).map_err(|e| format!("--out {}: {e}", dir.display()))?;
        std::fs::write(dir.join("outputs.f32"), &d.outputs).map_err(|e| format!("--out {}: {e}", dir.display()))?;
    }
    Ok(())
}
