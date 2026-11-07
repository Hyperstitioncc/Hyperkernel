use std::ops::Range;
use std::thread;

use super::spec::Spec;

pub const FIRST_FRAMES: usize = 57;
pub const NEW_FRAMES: usize = 40;
const CLIP_FRAMES: usize = 56;
const STRIDE: usize = 4;
const PINNED: (usize, usize) = (4, 4);
const BASE_STEP: usize = 8;
const RAY: usize = 6;
const WEIGHTS: usize = 4;

type Pose3 = [[f32; 4]; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Action {
    pub keys: [f32; 4],
    pub mouse: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct Camera {
    at: [f32; 5],
    poses: Vec<Pose>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub picks: Vec<usize>,
    pub poses: Vec<u8>,
    pub positions: Vec<u8>,
}

impl Pose {
    fn of(position: [f64; 3], pitch: f64, yaw: f64) -> Pose {
        let (sp, cp, sy, cy) = (pitch.to_radians().sin(), pitch.to_radians().cos(), yaw.to_radians().sin(), yaw.to_radians().cos());
        let rzy = [[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]];
        let r = rzy.map(|row| [row[1], -row[2], row[0]]);
        Pose { r, t: position.map(|v| v * 0.01) }
    }

    fn rounded(self) -> Pose3 {
        std::array::from_fn(|i| [self.r[i][0] as f32, self.r[i][1] as f32, self.r[i][2] as f32, self.t[i] as f32])
    }
}

impl Default for Camera {
    fn default() -> Camera {
        Camera { at: [0.0; 5], poses: vec![Pose::of([0.0; 3], 0.0, 0.0)] }
    }
}

impl Camera {
    pub fn act(&mut self, a: Action) {
        const OFFSET: f64 = 12.35;
        const DIAGONAL: f64 = 8.73;
        const SENSITIVITY: f64 = 15.0;
        const DEADZONE: f32 = 0.02;
        let [x, y, z, pitch, yaw] = self.at.map(f64::from);
        let turn = |m: f32| if m.abs() >= DEADZONE { SENSITIVITY * f64::from(m) } else { 0.0 };
        let new_pitch = pitch + turn(a.mouse[0]);
        let mut new_yaw = yaw + turn(a.mouse[1]);
        while new_yaw > 180.0 {
            new_yaw -= 360.0;
        }
        while new_yaw < -180.0 {
            new_yaw += 360.0;
        }
        let [w, s, l, r] = a.keys;
        let axis = |plus: f32, minus: f32| if plus > 0.5 && minus < 0.5 { OFFSET } else if minus > 0.5 && plus < 0.5 { -OFFSET } else { 0.0 };
        let (mut forward, mut right) = (axis(w, s), axis(r, l));
        if forward.abs() > 0.1 && right.abs() > 0.1 {
            (forward, right) = (forward.signum() * DIAGONAL, right.signum() * DIAGONAL);
        }
        let (sin, cos) = ((yaw + new_yaw) / 2.0).to_radians().sin_cos();
        self.at = [x + cos * forward - sin * right, y + sin * forward + cos * right, z, new_pitch, new_yaw].map(|v| v as f32);
        let [x, y, z, pitch, yaw] = self.at.map(f64::from);
        self.poses.push(Pose::of([x, y, z], pitch, yaw));
    }

    pub fn frames(&self) -> usize {
        self.poses.len()
    }

    pub fn chunk(&self, s: &Spec, c: usize) -> Chunk {
        let span = Span::of(c);
        assert!(self.frames() >= span.end && span.latents().end <= s.turns);
        let picks = if s.memory() > 0 { self.picks(&span) } else { Vec::new() };
        let mut frames: Vec<Pose3> = picks
            .iter()
            .enumerate()
            .map(|(slot, &frame)| {
                let aligned = (frame - 1) / STRIDE * STRIDE + 1;
                let mem = self.poses[aligned + 3].rounded().map(|row| row.map(f64::from));
                memory(&self.poses[span.base(slot)], &mem)
            })
            .collect();
        let clip: Vec<Pose3> = span.clip_frames().map(|f| self.poses[f].rounded()).collect();
        frames.extend(framewise(&clip));
        assert!(frames.len() == s.latents());
        let latents: Vec<usize> = picks.iter().map(|&f| latent(f)).chain(span.latents()).collect();
        let poses = frames.iter().flat_map(weights).flat_map(f32::to_le_bytes).collect();
        Chunk { picks: picks.iter().map(|&f| latent(f)).collect(), poses, positions: positions(s, &latents) }
    }

    fn picks(&self, span: &Span) -> Vec<usize> {
        let poses: Vec<Pose3> = self.poses.iter().map(|p| p.rounded()).collect();
        let cands = &poses[1..span.start];
        (0..=PINNED.0).map(|slot| if slot == PINNED.0 { PINNED.1 } else { 1 + best(&poses[span.base(slot)], cands) }).collect()
    }
}

struct Span {
    start: usize,
    end: usize,
}

impl Span {
    fn of(c: usize) -> Span {
        let end = FIRST_FRAMES + NEW_FRAMES * c;
        Span { start: end.saturating_sub(CLIP_FRAMES + usize::from(c == 0)), end }
    }

    fn base(&self, slot: usize) -> usize {
        self.end - 1 - BASE_STEP * slot
    }

    fn clip_frames(&self) -> impl Iterator<Item = usize> {
        (self.start + (self.end - self.start - 1) % STRIDE..self.end).step_by(STRIDE)
    }

    fn latents(&self) -> Range<usize> {
        let first = self.clip_frames().next().unwrap_or(self.end);
        latent(first)..latent(self.end - 1) + 1
    }
}

pub fn frames(c: usize) -> Range<usize> {
    let s = Span::of(c);
    s.start..s.end
}

pub fn kept(s: &Spec, c: usize) -> (usize, usize) {
    let fresh = s.latents() - s.split() / s.tokens;
    (Span::of(c).latents().end - fresh, fresh)
}

fn latent(f: usize) -> usize {
    if f == 0 { 0 } else { (f - 1) / STRIDE + 1 }
}

fn linspace(a: f32, b: f32, n: usize, i: usize) -> f32 {
    let step = (b - a) / (n - 1) as f32;
    if i < n / 2 { a + step * i as f32 } else { b - step * (n - 1 - i) as f32 }
}

fn turn(m: &Pose3, v: [f32; 3]) -> [f32; 3] {
    m.map(|row| row[2].mul_add(v[2], row[1].mul_add(v[1], row[0] * v[0])))
}

fn best(base: &Pose3, cands: &[Pose3]) -> usize {
    const SIDE: usize = 10;
    let (w, h, f) = (1280f32, 720f32, [640f32, 360f32]);
    let (near, far) = (0.1f32, 30f32);
    let points: Vec<[f32; 3]> = (0..SIDE * SIDE * SIDE)
        .map(|n| {
            let (x, y, z) = (linspace(-1.0, 1.0, SIDE, n / 100), linspace(-1.0, 1.0, SIDE, n / 10 % 10), linspace(near, far, SIDE, n % 10));
            let p = turn(base, [x * z, y * z, z]);
            [p[0] + base[0][3], p[1] + base[1][3], p[2] + base[2][3]]
        })
        .collect();
    let seen = |c: &Pose3| {
        let inv: Pose3 = std::array::from_fn(|i| [c[0][i], c[1][i], c[2][i], 0.0]);
        let back = turn(&inv, [c[0][3], c[1][3], c[2][3]]).map(|v| -v);
        points
            .iter()
            .filter(|&&p| {
                let q = turn(&inv, p);
                let [x, y, z] = [q[0] + back[0], q[1] + back[1], q[2] + back[2]];
                let zc = z.max(1e-6);
                let (u, v) = (x * f[0] / zc + w / 2.0, y * f[1] / zc + h / 2.0);
                z > near && z < far && (0.0..=w).contains(&u) && (0.0..=h).contains(&v)
            })
            .count()
    };
    let counts: Vec<usize> = parallel(cands.len(), |range| range.map(|i| seen(&cands[i])).collect::<Vec<_>>()).into_iter().flatten().collect();
    let most = counts.iter().copied().max().unwrap_or(0);
    counts.iter().position(|&n| n == most).unwrap_or(0)
}

fn framewise(poses: &[Pose3]) -> Vec<Pose3> {
    let inv = |p: &Pose3| -> Pose3 {
        let r: Pose3 = std::array::from_fn(|i| [p[0][i], p[1][i], p[2][i], 0.0]);
        let t = turn(&r, [p[0][3], p[1][3], p[2][3]]);
        std::array::from_fn(|i| [r[i][0], r[i][1], r[i][2], -t[i]])
    };
    let times = |a: &Pose3, b: &Pose3| -> Pose3 {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let s = a[i][2].mul_add(b[2][j], a[i][1].mul_add(b[1][j], a[i][0] * b[0][j]));
                if j == 3 { s + a[i][3] } else { s }
            })
        })
    };
    let eye: Pose3 = std::array::from_fn(|i| std::array::from_fn(|j| f32::from(u8::from(i == j))));
    let w2c = inv(&poses[0]);
    let mut rel: Vec<Pose3> = poses.iter().map(|p| times(&w2c, p)).collect();
    rel[0] = eye;
    let mut out: Vec<Pose3> = (0..rel.len()).map(|i| if i == 0 { eye } else { times(&inv(&rel[i - 1]), &rel[i]) }).collect();
    let norm = |p: &Pose3| (p[0][3] * p[0][3] + p[1][3] * p[1][3] + p[2][3] * p[2][3]).sqrt();
    let max = out.iter().map(norm).fold(0f32, f32::max);
    if max > 0.0 {
        out.iter_mut().for_each(|p| p.iter_mut().for_each(|row| row[3] /= max));
    }
    out
}

fn memory(base: &Pose, mem: &[[f64; 4]; 3]) -> Pose3 {
    let back: [f64; 3] = std::array::from_fn(|i| -(0..3).map(|k| base.r[k][i] * base.t[k]).sum::<f64>());
    let r: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| base.r[k][i] * mem[k][j]).sum()));
    let t: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| base.r[k][i] * mem[k][3]).sum::<f64>() + back[i]);
    let n = t.iter().map(|v| v * v).sum::<f64>().sqrt();
    let t = if n > 0.0 { t.map(|v| v / n) } else { t };
    std::array::from_fn(|i| [r[i][0] as f32, r[i][1] as f32, r[i][2] as f32, t[i] as f32])
}

fn weights(p: &Pose3) -> [f32; RAY * WEIGHTS] {
    std::array::from_fn(|n| {
        let (v, k) = (n / WEIGHTS, n % WEIGHTS);
        match (v < 3, k == 3) {
            (true, true) => p[v][3],
            (false, false) => p[v - 3][k],
            _ => 0.0,
        }
    })
}

fn positions(s: &Spec, latents: &[usize]) -> Vec<u8> {
    let (tokens, gw, pairs) = (s.tokens, s.grid[1], s.head / 2);
    let (t, hw) = (pairs - 2 * (pairs / 3), pairs / 3);
    latents
        .iter()
        .flat_map(|&l| (0..tokens).flat_map(move |n| (0..pairs).map(move |i| if i < t { l } else if i < t + hw { n / gw } else { n % gw })))
        .flat_map(|v| (v as i32).to_le_bytes())
        .collect()
}

fn parallel<R: Send>(n: usize, f: impl Fn(Range<usize>) -> R + Sync) -> Vec<R> {
    let per = n.div_ceil(thread::available_parallelism().map_or(1, |n| n.get())).max(1);
    thread::scope(|scope| {
        let parts: Vec<_> = (0..n)
            .step_by(per)
            .map(|a| {
                let f = &f;
                scope.spawn(move || f(a..(a + per).min(n)))
            })
            .collect();
        parts.into_iter().map(|p| p.join().expect("a host thread")).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mg3::spec;

    #[test]
    fn a_still_camera_picks_the_first_frame_and_its_clip_is_identities() {
        let mut cam = Camera::default();
        (0..FIRST_FRAMES + NEW_FRAMES).for_each(|_| cam.act(Action { keys: [0.0; 4], mouse: [0.0; 2] }));
        let span = Span::of(1);
        assert_eq!(cam.picks(&span), vec![1, 1, 1, 1, 4]);
        let clip: Vec<_> = span.clip_frames().map(|f| cam.poses[f].rounded()).collect();
        assert!(framewise(&clip).iter().all(|p| (0..3).all(|i| (0..4).all(|j| p[i][j] == f32::from(u8::from(i == j))))));
        assert_eq!(span.latents(), 11..25);
    }

    #[test]
    fn weights_and_positions() {
        let p: Pose3 = std::array::from_fn(|i| std::array::from_fn(|j| (4 * i + j) as f32));
        let w = weights(&p);
        assert_eq!((w[3], w[7], w[11], w[12..16].to_vec(), w[20..24].to_vec()), (3.0, 7.0, 11.0, vec![0.0, 1.0, 2.0, 0.0], vec![8.0, 9.0, 10.0, 0.0]));
        assert!(w[..3] == [0.0; 3]);
        let p = positions(&spec::chunk(), &[7]);
        let at = |row: usize, pair: usize| i32::from_le_bytes(p[(row * 64 + pair) * 4..][..4].try_into().expect("4 bytes"));
        assert_eq!((at(41, 0), at(41, 21), at(41, 22), at(41, 42), at(41, 43)), (7, 7, 1, 1, 1));
    }

    #[test]
    fn a_chunks_inputs_are_its_latents() {
        let s = spec::chunk();
        let mut cam = Camera::default();
        (0..FIRST_FRAMES - 1 + 2 * NEW_FRAMES).for_each(|_| cam.act(Action { keys: [1.0, 0.0, 0.0, 0.0], mouse: [0.0, 0.05] }));
        let k = cam.chunk(&s, 2);
        assert_eq!((k.picks.len(), k.poses.len(), k.positions.len()), (5, 19 * 24 * 4, 19 * 880 * 64 * 4));
        assert_eq!(kept(&s, 1), (15, 10));
    }
}
