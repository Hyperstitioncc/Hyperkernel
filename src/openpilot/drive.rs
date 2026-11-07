use std::time::Instant;

pub const SIDE: [usize; 2] = [128, 256];
pub const CHANNELS: usize = 24;
pub const WIDTH: usize = 512;
pub const DESIRE: usize = 200;
pub const TRAFFIC_LAID: usize = 8;
const CAMERAS: usize = 2;
const PLANES: usize = 6;
const IMG_BACK: usize = 4;
const FEATS_BACK: [usize; 8] = [32, 28, 24, 20, 16, 12, 8, 4];
pub const FRAME_BYTES: usize = CAMERAS * PLANES * SIDE[0] * SIDE[1];
pub const OUTPUTS: usize = 2576;
pub const WARMUP: usize = 20;
const HIDDEN: &str = "hidden_state";
const PAD: &str = "pad";

pub const SLICES: [(&str, usize); 14] = [
    ("meta", 55),
    ("desire_pred", 32),
    ("pose", 12),
    ("wide_from_device_euler", 6),
    ("road_transform", 12),
    ("lane_lines", 528),
    ("lane_lines_prob", 8),
    ("road_edges", 264),
    ("lead", 144),
    ("lead_prob", 3),
    (HIDDEN, 512),
    ("plan", 990),
    ("desire_state", 8),
    (PAD, 2),
];

pub trait Step {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String>;
    fn launch(&mut self) -> Result<(), String>;
    fn read_all(&self, items: &[(&str, usize)]) -> Result<Vec<Vec<u8>>, String>;
}

pub struct Drive {
    pub outputs: Vec<u8>,
    pub lines: String,
}

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn widened(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(2)
        .flat_map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).to_le_bytes())
        .collect()
}

pub fn img(frames: &[u8], t: usize) -> Vec<u8> {
    let [h, w] = SIDE;
    let at = |f: Option<usize>, cam: usize, plane: usize, y: usize, x: usize| {
        f.map_or(0, |f| frames[f * FRAME_BYTES + ((cam * PLANES + plane) * h + y) * w + x])
    };
    let mut b = Vec::with_capacity(h * w * CHANNELS * 2);
    for y in 0..h {
        for x in 0..w {
            for cam in 0..CAMERAS {
                for f in [t.checked_sub(IMG_BACK), Some(t)] {
                    for plane in 0..PLANES {
                        b.extend_from_slice(&bf16(f32::from(at(f, cam, plane, y, x))).to_le_bytes());
                    }
                }
            }
        }
    }
    b
}

pub fn feats(hidden: &[Vec<u8>], t: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(FEATS_BACK.len() * WIDTH * 2);
    for back in FEATS_BACK {
        match t.checked_sub(back) {
            Some(s) => b.extend_from_slice(&hidden[s]),
            None => b.resize(b.len() + WIDTH * 2, 0),
        }
    }
    b
}

pub fn traffic() -> Vec<u8> {
    let mut b = vec![0; TRAFFIC_LAID * 2];
    b[..2].copy_from_slice(&bf16(1.0).to_le_bytes());
    b
}

pub fn reads() -> Vec<(&'static str, usize)> {
    SLICES
        .iter()
        .filter(|(name, _)| *name != PAD)
        .map(|&(name, n)| (name, if name == HIDDEN { n * 2 } else { n * 4 }))
        .collect()
}

pub fn row(got: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    let mut got = got.iter();
    let mut row = Vec::with_capacity(OUTPUTS * 4);
    for (name, n) in SLICES {
        match name {
            PAD => row.resize(row.len() + n * 4, 0),
            HIDDEN => row.extend(widened(got.next().ok_or("openpilot: an output short")?)),
            _ => row.extend_from_slice(got.next().ok_or("openpilot: an output short")?),
        }
    }
    Ok(row)
}

pub fn drive(frames: &[u8], n: usize, step: &mut impl Step) -> Result<Drive, String> {
    if n == 0 {
        return Err("no frames to drive".into());
    }
    if frames.len() < n * FRAME_BYTES {
        return Err(format!("{} bytes of frames, under {n} frames of {FRAME_BYTES}", frames.len()));
    }
    let (desire, road, reads) = (vec![0u8; DESIRE * 2], traffic(), reads());
    let hidden_at = reads.iter().position(|&(name, _)| name == HIDDEN).expect("a hidden state read");
    let (mut hidden, mut outputs, mut walls, mut lines) =
        (Vec::with_capacity(n), Vec::with_capacity(n * OUTPUTS * 4), Vec::with_capacity(n), String::new());
    for t in 0..n {
        let (i, f) = (img(frames, t), feats(&hidden, t));
        let t0 = Instant::now();
        step.write_all(&[("img", &i), ("feats", &f), ("desire", &desire), ("traffic", &road)])?;
        let t1 = Instant::now();
        step.launch()?;
        let t2 = Instant::now();
        let got = step.read_all(&reads)?;
        let t3 = Instant::now();
        outputs.extend(row(&got)?);
        hidden.push(got[hidden_at].clone());
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        lines += &format!(
            "FRAME f={t} wall_ms={:.3} write_ms={:.3} launch_ms={:.3} read_ms={:.3}\n",
            ms(t0, t3),
            ms(t0, t1),
            ms(t1, t2),
            ms(t2, t3)
        );
        walls.push(ms(t0, t3));
    }
    lines += &summary(n, &walls);
    Ok(Drive { outputs, lines })
}

fn summary(n: usize, walls: &[f64]) -> String {
    let warmup = WARMUP.min(n - 1);
    let mut past: Vec<f64> = walls.iter().skip(warmup).copied().collect();
    past.sort_by(f64::total_cmp);
    let q = |p: f64| past[((past.len() as f64 * p) as usize).min(past.len() - 1)];
    format!(
        "DRIVE frames={n} warmup={warmup} median_ms={:.3} p99_ms={:.3} max_ms={:.3} frames_per_s={:.1}\n",
        q(0.5),
        q(0.99),
        q(1.0),
        1e3 / q(0.5)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn img_orders_as_model_cat() {
        let mut d = vec![0u8; 5 * FRAME_BYTES];
        let plane = SIDE[0] * SIDE[1];
        let at = |f: usize, cam: usize, p: usize, y: usize, x: usize| {
            f * FRAME_BYTES + (cam * 6 + p) * plane + y * SIDE[1] + x
        };
        d[at(4, 1, 5, 3, 7)] = 200;
        d[at(0, 0, 2, 3, 7)] = 9;
        let b = img(&d, 4);
        let word = |ch: usize| {
            let o = (3 * SIDE[1] + 7) * 24 * 2 + ch * 2;
            f32::from_bits(u32::from(u16::from_le_bytes([b[o], b[o + 1]])) << 16)
        };
        assert_eq!(word(23), 200.0);
        assert_eq!(word(2), 9.0);
        assert_eq!(img(&d, 3).len(), SIDE[0] * SIDE[1] * CHANNELS * 2);
        let first = img(&d, 0);
        let o = (3 * SIDE[1] + 7) * 24 * 2 + 2 * 2;
        assert_eq!((first[o], first[o + 1]), (0, 0));
    }

    #[test]
    fn feats_rows() {
        let hidden: Vec<Vec<u8>> = (0..40).map(|s| vec![s as u8; WIDTH * 2]).collect();
        let b = feats(&hidden, 30);
        assert_eq!(b.len(), 8 * WIDTH * 2);
        assert_eq!(b[0], 0);
        assert_eq!(b[WIDTH * 2], 2);
        assert_eq!(b[7 * WIDTH * 2], 26);
    }

    struct Echo {
        fed: Vec<Vec<u8>>,
        launches: usize,
    }

    impl Step for Echo {
        fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
            assert_eq!(items.iter().map(|i| i.0).collect::<Vec<_>>(), ["img", "feats", "desire", "traffic"]);
            self.fed.push(items[1].1.to_vec());
            Ok(())
        }

        fn launch(&mut self) -> Result<(), String> {
            self.launches += 1;
            Ok(())
        }

        fn read_all(&self, items: &[(&str, usize)]) -> Result<Vec<Vec<u8>>, String> {
            let step = self.launches as f32;
            Ok(items
                .iter()
                .map(|&(name, n)| match name {
                    HIDDEN => (0..n / 2).flat_map(|_| bf16(step).to_le_bytes()).collect(),
                    _ => (0..n / 4).flat_map(|_| (-step).to_le_bytes()).collect(),
                })
                .collect())
        }
    }

    #[test]
    fn a_drive_feeds_each_steps_hidden_state_back_and_writes_the_graphs_rows() {
        let n = 40;
        let mut echo = Echo { fed: Vec::new(), launches: 0 };
        let d = drive(&vec![7u8; n * FRAME_BYTES], n, &mut echo).unwrap();
        assert_eq!(d.outputs.len(), n * OUTPUTS * 4);
        assert_eq!(d.lines.lines().filter(|l| l.starts_with("FRAME ")).count(), n);
        assert!(d.lines.lines().last().unwrap().starts_with("DRIVE frames=40"));
        let word = |f: usize, i: usize| {
            f32::from_le_bytes(d.outputs[(f * OUTPUTS + i) * 4..][..4].try_into().unwrap())
        };
        let hidden = SLICES.iter().take_while(|s| s.0 != HIDDEN).map(|s| s.1).sum::<usize>();
        assert_eq!(word(5, 0), -6.0);
        assert_eq!(word(5, hidden), 6.0);
        assert_eq!(word(5, OUTPUTS - 1), 0.0);
        assert!(echo.fed[3].iter().all(|&b| b == 0));
        assert_eq!(
            f32::from_bits(u32::from(u16::from_le_bytes([echo.fed[32][0], echo.fed[32][1]])) << 16),
            1.0
        );
        assert!(drive(&[0u8; 10], 1, &mut echo).is_err());
        assert!(drive(&[], 0, &mut echo).is_err());
        let short = drive(&vec![7u8; 3 * FRAME_BYTES], 3, &mut echo).unwrap();
        let last = short.lines.lines().last().unwrap();
        assert!(last.starts_with("DRIVE frames=3 warmup=2 ") && !last.contains("NaN"), "{last}");
    }
}
