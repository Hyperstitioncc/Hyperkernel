use std::path::{Path, PathBuf};

use crate::act::tick::{self, IMAGE_BYTES};
use crate::act::{Act, Device};
use crate::bundle::Bundle;
use crate::gpu::{Pinned, cuda, load, ranks};
use crate::weights::Checkpoint;

pub const GPUS: usize = 1;

pub struct Flags {
    pub images: PathBuf,
    pub states: Option<PathBuf>,
    pub count: Option<usize>,
    pub out: Option<PathBuf>,
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let images =
        std::fs::read(&flags.images).map_err(|e| format!("--images {}: {e}", flags.images.display()))?;
    let states = match &flags.states {
        Some(p) => std::fs::read(p).map_err(|e| format!("--states {}: {e}", p.display()))?,
        None => Vec::new(),
    };
    let n = flags.count.unwrap_or(images.len() / IMAGE_BYTES);
    let bundle = Bundle::<Act>::read(kernels, 0)?;
    cuda::visible(GPUS)?;
    let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric)?;
    let loaded = load::load(&bundle, &ranks, Some(&Checkpoint::open(weights)?), &[])?;
    let stage = Pinned::new(IMAGE_BYTES)?;
    let t = tick::ticks(&images, &states, n, &mut Device { bundle: &bundle, loaded, stage })?;
    print!("{}", t.lines);
    if let Some(dir) = flags.out {
        std::fs::create_dir_all(&dir).map_err(|e| format!("--out {}: {e}", dir.display()))?;
        std::fs::write(dir.join("actions.f32"), &t.actions)
            .map_err(|e| format!("--out {}: {e}", dir.display()))?;
    }
    Ok(())
}
