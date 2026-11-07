use std::time::Instant;

pub const IMAGE: [usize; 2] = [480, 640];
pub const COLOURS: usize = 3;
pub const JOINTS: usize = 14;
pub const STATE_COLS: usize = 16;
pub const CHUNK: usize = 100;
pub const IMAGE_BYTES: usize = IMAGE[0] * IMAGE[1] * COLOURS;
pub const STATE_BYTES: usize = STATE_COLS * 4;
pub const ACTIONS_BYTES: usize = CHUNK * JOINTS * 4;
pub const WARMUP: usize = 20;

pub trait Step {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String>;
    fn launch(&mut self) -> Result<(), String>;
    fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String>;
}

pub struct Ticks {
    pub actions: Vec<u8>,
    pub lines: String,
}

pub fn state(states: &[u8], t: usize) -> Vec<u8> {
    let mut b = vec![0u8; STATE_BYTES];
    if !states.is_empty() {
        let rows = states.len() / (JOINTS * 4);
        let at = (t % rows) * JOINTS * 4;
        b[..JOINTS * 4].copy_from_slice(&states[at..at + JOINTS * 4]);
    }
    b
}

pub fn ticks(images: &[u8], states: &[u8], n: usize, step: &mut impl Step) -> Result<Ticks, String> {
    if n == 0 {
        return Err("no ticks to run".into());
    }
    let frames = images.len() / IMAGE_BYTES;
    if frames == 0 || !images.len().is_multiple_of(IMAGE_BYTES) {
        return Err(format!(
            "{} bytes of images, not whole {}x{}x{} u8 frames",
            images.len(),
            IMAGE[0],
            IMAGE[1],
            COLOURS
        ));
    }
    if !states.len().is_multiple_of(JOINTS * 4) {
        return Err(format!("{} bytes of states, not whole rows of {JOINTS} f32", states.len()));
    }
    let (mut actions, mut walls, mut lines) =
        (Vec::with_capacity(n * ACTIONS_BYTES), Vec::with_capacity(n), String::new());
    for t in 0..n {
        let f = t % frames;
        let (image, s) = (&images[f * IMAGE_BYTES..(f + 1) * IMAGE_BYTES], state(states, t));
        let t0 = Instant::now();
        step.write_all(&[("image", image), ("state", &s)])?;
        let t1 = Instant::now();
        step.launch()?;
        let t2 = Instant::now();
        let a = step.read("actions", ACTIONS_BYTES)?;
        let t3 = Instant::now();
        actions.extend(a);
        let us = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e6;
        lines += &format!(
            "TICK t={t} frame={f} wall_us={:.1} write_us={:.1} launch_us={:.1} read_us={:.1}\n",
            us(t0, t3),
            us(t0, t1),
            us(t1, t2),
            us(t2, t3)
        );
        walls.push(us(t0, t3));
    }
    lines += &summary(n, &walls);
    Ok(Ticks { actions, lines })
}

fn summary(n: usize, walls: &[f64]) -> String {
    let warmup = WARMUP.min(n - 1);
    let mut past: Vec<f64> = walls.iter().skip(warmup).copied().collect();
    past.sort_by(f64::total_cmp);
    let q = |p: f64| past[((past.len() as f64 * p) as usize).min(past.len() - 1)];
    format!(
        "ACT ticks={n} warmup={warmup} median_us={:.1} p99_us={:.1} max_us={:.1} ticks_per_s={:.1}\n",
        q(0.5),
        q(0.99),
        q(1.0),
        1e6 / q(0.5)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo {
        fed: Vec<(u8, Vec<u8>)>,
        launches: usize,
    }

    impl Step for Echo {
        fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
            assert_eq!(
                items.iter().map(|i| (i.0, i.1.len())).collect::<Vec<_>>(),
                [("image", IMAGE_BYTES), ("state", STATE_BYTES)]
            );
            self.fed.push((items[0].1[0], items[1].1.to_vec()));
            Ok(())
        }

        fn launch(&mut self) -> Result<(), String> {
            self.launches += 1;
            Ok(())
        }

        fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String> {
            assert_eq!((name, bytes), ("actions", ACTIONS_BYTES));
            Ok((0..bytes / 4).flat_map(|_| (self.launches as f32).to_le_bytes()).collect())
        }
    }

    fn word(b: &[u8], i: usize) -> f32 {
        f32::from_le_bytes(b[i * 4..][..4].try_into().unwrap())
    }

    #[test]
    fn ticks_cycle_the_frames_pad_the_state_and_keep_each_chunk() {
        let images: Vec<u8> = (0..3u8).flat_map(|f| vec![f + 1; IMAGE_BYTES]).collect();
        let states: Vec<u8> = (0..2)
            .flat_map(|r| (0..JOINTS).flat_map(move |j| (r as f32 * 100.0 + j as f32).to_le_bytes()))
            .collect();
        let mut echo = Echo { fed: Vec::new(), launches: 0 };
        let t = ticks(&images, &states, 25, &mut echo).unwrap();
        assert_eq!(echo.fed.iter().map(|f| f.0).take(4).collect::<Vec<_>>(), [1, 2, 3, 1]);
        let s = &echo.fed[3].1;
        assert_eq!((word(s, 0), word(s, 13), word(s, 14), word(s, 15)), (100.0, 113.0, 0.0, 0.0));
        assert_eq!(t.actions.len(), 25 * ACTIONS_BYTES);
        assert_eq!(word(&t.actions, 24 * CHUNK * JOINTS), 25.0);
        assert_eq!(t.lines.lines().filter(|l| l.starts_with("TICK ")).count(), 25);
        assert!(t.lines.lines().last().unwrap().starts_with("ACT ticks=25 warmup=20 "));
    }

    #[test]
    fn no_states_feed_zeros_and_bad_inputs_are_refused() {
        let mut echo = Echo { fed: Vec::new(), launches: 0 };
        ticks(&vec![9u8; IMAGE_BYTES], &[], 2, &mut echo).unwrap();
        assert!(echo.fed[1].1.iter().all(|&b| b == 0));
        assert!(ticks(&vec![0u8; IMAGE_BYTES - 1], &[], 1, &mut echo).is_err());
        assert!(ticks(&vec![0u8; IMAGE_BYTES], &[0u8; 10], 1, &mut echo).is_err());
        assert!(ticks(&vec![0u8; IMAGE_BYTES], &[], 0, &mut echo).is_err());
        let one = ticks(&vec![0u8; IMAGE_BYTES], &[], 1, &mut echo).unwrap();
        assert!(one.lines.lines().last().unwrap().starts_with("ACT ticks=1 warmup=0 "));
    }
}
