use std::time::Instant;

pub const CAMERAS: usize = 3;
pub const SIDE: usize = 224;
pub const IMAGE_FLOATS: usize = CAMERAS * 3 * SIDE * SIDE;
pub const PROMPT: usize = 200;
pub const TOKENS: usize = 257;
pub const EOS: i32 = 1;
pub const WARMUP: usize = 3;

pub struct Observation<'a> {
    pub images: &'a [f32],
    pub ids: &'a [i32],
    pub lens: i32,
}

pub trait Step {
    fn chunk(&mut self, o: &Observation<'_>) -> Result<Vec<i32>, String>;
}

pub fn ended(tokens: &[i32], fixed: Option<usize>) -> &[i32] {
    match fixed {
        Some(n) => &tokens[..n.min(tokens.len())],
        None => &tokens[..tokens.iter().position(|&t| t == EOS).map_or(tokens.len(), |k| k + 1)],
    }
}

pub struct Chunks {
    pub tokens: Vec<Vec<i32>>,
    pub lines: String,
}

pub fn chunks(o: &Observation<'_>, n: usize, fixed: Option<usize>, step: &mut impl Step) -> Result<Chunks, String> {
    if n == 0 {
        return Err("no chunks to run".into());
    }
    if o.images.len() != IMAGE_FLOATS || o.ids.len() != PROMPT {
        return Err(format!(
            "an observation of {} image floats and {} ids, not {IMAGE_FLOATS} ({CAMERAS} cameras of 3 x {SIDE} x {SIDE}) and {PROMPT}",
            o.images.len(),
            o.ids.len()
        ));
    }
    let (mut tokens, mut walls, mut lines) = (Vec::with_capacity(n), Vec::with_capacity(n), String::new());
    for k in 0..n {
        let t0 = Instant::now();
        let all = step.chunk(o)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let got = ended(&all, fixed).to_vec();
        lines += &format!("CHUNK run={k} ms={ms:.3} tokens={} ids={got:?}\n", got.len());
        walls.push(ms);
        tokens.push(got);
    }
    let warmup = WARMUP.min(n - 1);
    let mut past: Vec<f64> = walls[warmup..].to_vec();
    past.sort_by(f64::total_cmp);
    let q = |p: f64| past[((past.len() as f64 * p) as usize).min(past.len() - 1)];
    lines += &format!("PI0FAST chunks={n} warmup={warmup} median_ms={:.3} p99_ms={:.3} max_ms={:.3}\n", q(0.5), q(0.99), q(1.0));
    Ok(Chunks { tokens, lines })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        calls: usize,
    }

    impl Step for Fake {
        fn chunk(&mut self, o: &Observation<'_>) -> Result<Vec<i32>, String> {
            assert_eq!((o.images.len(), o.ids.len(), o.lens), (IMAGE_FLOATS, PROMPT, 7));
            self.calls += 1;
            let mut t: Vec<i32> = (10..10 + TOKENS as i32).collect();
            t[5] = EOS;
            Ok(t)
        }
    }

    #[test]
    fn a_chunk_ends_at_eos_or_its_fixed_count_and_warmup_is_left_out() {
        let (images, ids) = (vec![0.0f32; IMAGE_FLOATS], vec![0i32; PROMPT]);
        let o = Observation { images: &images, ids: &ids, lens: 7 };
        let mut f = Fake { calls: 0 };
        let c = chunks(&o, 5, None, &mut f).unwrap();
        assert_eq!(f.calls, 5);
        assert_eq!(c.tokens[0], [10, 11, 12, 13, 14, EOS]);
        assert!(c.lines.lines().last().unwrap().starts_with("PI0FAST chunks=5 warmup=3 "));
        let fixed = chunks(&o, 1, Some(60), &mut f).unwrap();
        assert_eq!(fixed.tokens[0].len(), 60);
        assert!(fixed.lines.lines().last().unwrap().starts_with("PI0FAST chunks=1 warmup=0 "));
        assert!(chunks(&Observation { images: &images[1..], ids: &ids, lens: 7 }, 1, None, &mut f).is_err());
        assert!(chunks(&o, 0, None, &mut f).is_err());
        assert_eq!(ended(&[3, 4], None), [3, 4]);
    }
}
