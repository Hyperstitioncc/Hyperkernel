use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};
use std::thread::{self, JoinHandle};

use super::camera::{Action, NEW_FRAMES};

pub const HEIGHT: usize = 704;
pub const WIDTH: usize = 1280;
const PATCH: usize = 2;
const COLOURS: usize = 3;
const ROW_VALUES: usize = COLOURS * PATCH * PATCH;

pub const fn chunk_bytes() -> usize {
    NEW_FRAMES * (HEIGHT / PATCH) * (WIDTH / PATCH) * ROW_VALUES * 2
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Demo {
    pub actions: Option<Vec<Action>>,
    pub frames: Option<PathBuf>,
    pub timing: Option<PathBuf>,
    pub context: Option<PathBuf>,
    pub noise: Option<PathBuf>,
}

pub fn actions(text: &str) -> Result<Vec<Action>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let v = line.split_whitespace().map(str::parse::<f32>).collect::<Result<Vec<f32>, _>>().map_err(|e| format!("line {}: {e}", i + 1))?;
        let &[w, s, a, d, pitch, yaw] = v.as_slice() else {
            return Err(format!("line {}: {} values, a frame's are w s a d pitch yaw", i + 1, v.len()));
        };
        out.push(Action { keys: [w, s, a, d], mouse: [pitch, yaw] });
    }
    Ok(out)
}

pub fn rgb(bytes: &[u8]) -> Vec<u8> {
    let (h, w) = (HEIGHT / PATCH, WIDTH / PATCH);
    let frame = h * w * ROW_VALUES * 2;
    let mut out = vec![0u8; bytes.len() / frame * HEIGHT * WIDTH * COLOURS];
    let value = |b: &[u8]| {
        let v = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
        ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8
    };
    for (f, frame_bytes) in bytes.chunks_exact(frame).enumerate() {
        let image = f * HEIGHT * WIDTH * COLOURS;
        for (r, row) in frame_bytes.chunks_exact(ROW_VALUES * 2).enumerate() {
            let (py, px) = (r / w * PATCH, r % w * PATCH);
            for x in 0..PATCH {
                for y in 0..PATCH {
                    let pixel = image + ((py + y) * WIDTH + px + x) * COLOURS;
                    for c in 0..COLOURS {
                        out[pixel + c] = value(&row[(c * PATCH * PATCH + x * PATCH + y) * 2..]);
                    }
                }
            }
        }
    }
    out
}

pub struct Frames {
    send: Sender<Vec<u8>>,
    thread: JoinHandle<io::Result<()>>,
}

impl Frames {
    pub fn open(path: &Path) -> Result<Frames, String> {
        let file = File::create(path).map_err(|e| format!("--frames-out {}: {e}", path.display()))?;
        let (send, take) = channel::<Vec<u8>>();
        let thread = thread::spawn(move || {
            let mut out = BufWriter::new(file);
            for chunk in take {
                out.write_all(&rgb(&chunk))?;
            }
            out.flush()
        });
        Ok(Frames { send, thread })
    }

    pub fn push(&self, chunk: Vec<u8>) {
        let _ = self.send.send(chunk);
    }

    pub fn close(self) -> Result<(), String> {
        drop(self.send);
        let written = self.thread.join().map_err(|_| "its writer panicked".to_string())?;
        written.map_err(|e| format!("--frames-out: {e}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Times {
    pub chunk: usize,
    pub start_ms: f64,
    pub lead_ms: f64,
    pub frames_ms: Option<f64>,
    pub read_ms: Option<f64>,
}

impl Times {
    fn line(&self) -> String {
        let opt = |v: Option<f64>| v.map_or("null".to_owned(), |t| format!("{t:.3}"));
        format!(
            "{{\"chunk\":{},\"frames\":{NEW_FRAMES},\"start_ms\":{:.3},\"lead_ms\":{:.3},\"frames_ms\":{},\"read_ms\":{}}}\n",
            self.chunk,
            self.start_ms,
            self.lead_ms,
            opt(self.frames_ms),
            opt(self.read_ms)
        )
    }
}

pub fn write_times(path: &Path, times: &[Times]) -> Result<(), String> {
    let text: String = times.iter().map(Times::line).collect();
    std::fs::write(path, text).map_err(|e| format!("--timing-out {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{COLOURS, HEIGHT, PATCH, ROW_VALUES, Times, WIDTH, actions, chunk_bytes, rgb};
    use crate::mg3::camera::Action;

    #[test]
    fn a_route_parses() {
        let a = actions("# walk\n1 0 0 0 0 0.05\n\n0 0 1 0 -0.1 0 # strafe\n").unwrap();
        assert_eq!(a, vec![Action { keys: [1.0, 0.0, 0.0, 0.0], mouse: [0.0, 0.05] }, Action { keys: [0.0, 0.0, 1.0, 0.0], mouse: [-0.1, 0.0] }]);
        assert_eq!(actions("1 0 0\n"), Err("line 1: 3 values, a frame's are w s a d pitch yaw".to_owned()));
    }

    #[test]
    fn a_chunks_bytes() {
        assert_eq!(chunk_bytes(), 40 * 352 * 640 * 12 * 2);
    }

    #[test]
    fn a_patch_lands_as_unpatchify_lays_it() {
        let bf16 = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
        let frame = (HEIGHT / PATCH) * (WIDTH / PATCH) * ROW_VALUES * 2;
        let mut bytes = vec![0u8; frame];
        bytes.chunks_exact_mut(2).for_each(|b| b.copy_from_slice(&bf16(-1.0)));
        let row = (WIDTH / PATCH + 1) * ROW_VALUES * 2;
        bytes[row + 5 * 2..row + 6 * 2].copy_from_slice(&bf16(3.0));
        let out = rgb(&bytes);
        assert_eq!(out.len(), HEIGHT * WIDTH * COLOURS);
        assert_eq!(out[((PATCH + 1) * WIDTH + PATCH) * COLOURS + 1], 255);
        assert_eq!(out.iter().filter(|&&v| v != 0).count(), 1);
    }

    #[test]
    fn a_times_line() {
        let t = Times { chunk: 2, start_ms: 1.5, lead_ms: 2.0, frames_ms: None, read_ms: None };
        assert_eq!(t.line(), "{\"chunk\":2,\"frames\":40,\"start_ms\":1.500,\"lead_ms\":2.000,\"frames_ms\":null,\"read_ms\":null}\n");
    }
}
