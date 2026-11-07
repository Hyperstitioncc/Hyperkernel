use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use clap::Args;

use crate::gpu::cuda;
use crate::gpu::ranks::Ranks;
use crate::mg3::demo::{self, Demo};
use crate::mg3::gpu::{Kernel, Pipe, Seat};
use crate::mg3::link::Link;
use crate::mg3::run::{self, Specs};
use crate::weights::Checkpoint;

pub const GPUS: usize = 2;

const LINK: &str = "MG3_LINK";
const VISIBLE: &str = "CUDA_VISIBLE_DEVICES";
const DIAL: Duration = Duration::from_secs(1800);

#[derive(Args)]
pub struct Flags {
    #[arg(long, default_value_t = 4, help = "Chunks of 40 new frames to generate after the first clip of 57")]
    steps: usize,
    #[arg(long, help = "The first frame's latent: BF16 token rows [default: zeros]")]
    image_latent: Option<PathBuf>,
    #[arg(long, help = "The text prompt's encoder rows, BF16 [default: zeros]")]
    context: Option<PathBuf>,
    #[arg(long, help = "A directory of noise-C.bin, BF16, per chunk C from 0 [default: seeded normal draws]")]
    noise: Option<PathBuf>,
    #[arg(long, help = "A text file of one line per frame: w s a d pitch yaw [default: walking forward, turning slowly]")]
    actions: Option<PathBuf>,
    #[arg(long, help = "A file to write the frames into, raw 1280x704 RGB u8 (frames are written only with this)")]
    frames_out: Option<PathBuf>,
    #[arg(long, help = "A file to write per-chunk timings into, one JSON line per chunk")]
    timing_out: Option<PathBuf>,
}

pub fn bundle(_: &Flags) -> &'static str {
    "mg3"
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    match std::env::var_os(LINK) {
        Some(path) => vae(kernels, weights, Path::new(&path)),
        None => dit(kernels, weights, &flags),
    }
}

fn vae(kernels: &Path, weights: &Path, path: &Path) -> Result<(), String> {
    let bundle = Kernel::open(&kernels.join("mg3vae"))?;
    let ranks = Ranks::join(0, bundle.reference.kernel.symmetric)?;
    let kernel = Kernel::boot(bundle, &ranks, &Checkpoint::open(weights)?)?;
    let start = Instant::now();
    let link = loop {
        match UnixStream::connect(path) {
            Ok(s) => break Link(s),
            Err(e) if start.elapsed() > DIAL => return Err(format!("{}: {e}", path.display())),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    link.serve(|ask, payload| kernel.answer(ask, payload))
}

fn dit(kernels: &Path, weights: &Path, flags: &Flags) -> Result<(), String> {
    let demo = asks(flags)?;
    let image = flags.image_latent.as_ref().map(|f| std::fs::read(f).map_err(|e| format!("--image-latent {}: {e}", f.display()))).transpose()?;
    cuda::visible(GPUS)?;
    let gpu = second(std::env::var(VISIBLE).ok().as_deref())?;
    let path = std::env::temp_dir().join(format!("mg3-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut child = spawn(&path, &gpu)?;
    let (chunk, first) = (Kernel::open(&kernels.join("mg3chunk"))?, Kernel::open(&kernels.join("mg3first"))?);
    let symmetric = chunk.reference.kernel.symmetric.max(first.reference.kernel.symmetric);
    let ranks = Ranks::join(0, symmetric)?;
    let weights = Checkpoint::open(weights)?;
    let kernels = vec![Kernel::boot(chunk, &ranks, &weights)?, Kernel::boot(first, &ranks, &weights)?];
    let there = accept(&listener, &mut child)?;
    let _ = std::fs::remove_file(&path);
    let seats = vec![Seat::Here { kernel: 0, code: 0 }, Seat::Here { kernel: 0, code: 1 }, Seat::Here { kernel: 1, code: 0 }, Seat::Here { kernel: 1, code: 1 }, Seat::There { code: 0 }];
    let mut pipe = Pipe { kernels, there, seats };
    let (c, f) = (pipe.kernels[0].facts().model.clone(), pipe.kernels[1].facts().model.clone());
    let lines = run::run(&mut pipe, Specs { chunk: &c, first: &f }, image.as_deref(), flags.steps, &demo)?;
    print!("{lines}");
    drop(pipe);
    let status = child.wait().map_err(|e| format!("the VAE's process: {e}"))?;
    match status.success() {
        true => Ok(()),
        false => Err(format!("the VAE's process ended: {status}")),
    }
}

fn asks(flags: &Flags) -> Result<Demo, String> {
    let actions = flags.actions.as_ref().map(|f| std::fs::read_to_string(f).map_err(|e| e.to_string()).and_then(|t| demo::actions(&t)).map_err(|e| format!("--actions {}: {e}", f.display()))).transpose()?;
    Ok(Demo { actions, frames: flags.frames_out.clone(), timing: flags.timing_out.clone(), context: flags.context.clone(), noise: flags.noise.clone() })
}

pub fn second(visible: Option<&str>) -> Result<String, String> {
    match visible {
        None => Ok("1".to_string()),
        Some(v) => v.split(',').nth(1).map(|g| g.trim().to_string()).ok_or_else(|| format!("{VISIBLE}={v}: the VAE runs on the second visible GPU")),
    }
}

fn spawn(path: &Path, gpu: &str) -> Result<Child, String> {
    let exe = std::env::current_exe().map_err(|e| format!("this executable: {e}"))?;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    Command::new(exe).args(args).env(VISIBLE, gpu).env(LINK, path).spawn().map_err(|e| format!("the VAE's process: {e}"))
}

fn accept(listener: &UnixListener, child: &mut Child) -> Result<Link, String> {
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    loop {
        match listener.accept() {
            Ok((s, _)) => {
                s.set_nonblocking(false).map_err(|e| e.to_string())?;
                return Ok(Link(s));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(format!("the VAE's link: {e}")),
        }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!("the VAE's process ended before its link: {status}"));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vae_runs_on_the_second_visible_gpu() {
        assert_eq!(second(None).unwrap(), "1");
        assert_eq!(second(Some("4,5,6")).unwrap(), "5");
        assert!(second(Some("3")).unwrap_err().contains("second visible GPU"));
    }
}
