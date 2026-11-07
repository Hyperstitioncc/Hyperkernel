use std::path::{Path, PathBuf};

use crate::bundle::Bundle;
use crate::gpu::{cuda, load, ranks};
use crate::pi0::chunk::{self, Observation};
use crate::pi0::{Device, Pi0};
use crate::weights::Checkpoint;

pub const GPUS: usize = 1;
pub const FIXED: usize = 60;
pub const COUNT: usize = 13;

pub struct Flags {
    pub images: PathBuf,
    pub ids: PathBuf,
    pub lens: Option<i32>,
    pub lens_file: Option<PathBuf>,
    pub count: usize,
    pub fixed60: bool,
    pub out: Option<PathBuf>,
}

fn read(p: &Path, what: &str) -> Result<Vec<u8>, String> {
    std::fs::read(p).map_err(|e| format!("--{what} {}: {e}", p.display()))
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let images: Vec<f32> = read(&flags.images, "images")?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let ids: Vec<i32> = read(&flags.ids, "ids")?.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    let lens = match (flags.lens, &flags.lens_file) {
        (Some(n), _) => n,
        (None, Some(p)) => read(p, "lens-file")?.first_chunk::<4>().map(|b| i32::from_le_bytes(*b)).ok_or("--lens-file: under 4 bytes")?,
        (None, None) => return Err("--lens N or --lens-file FILE".into()),
    };
    let bundle = Bundle::<Pi0>::read(kernels, 0)?;
    cuda::visible(GPUS)?;
    let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric)?;
    let loaded = load::load(&bundle, &ranks, Some(&Checkpoint::open(weights)?), &[])?;
    let mut dev = Device::boot(&bundle, loaded)?;
    let o = Observation { images: &images, ids: &ids, lens };
    let c = chunk::chunks(&o, flags.count, flags.fixed60.then_some(FIXED), &mut dev)?;
    dev.close()?;
    print!("{}", c.lines);
    if let (Some(p), Some(last)) = (flags.out, c.tokens.last()) {
        let bytes: Vec<u8> = last.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&p, bytes).map_err(|e| format!("--out {}: {e}", p.display()))?;
    }
    Ok(())
}
