pub mod cli;
pub mod fold;

use serde::Deserialize;

pub use crate::bundle::{Port, Ports};

use crate::bundle::{Bundle, Served};
use crate::gpu::load::Loaded;
use crate::gpu::ranks::Ranks;
use fold::Prog;

#[derive(Deserialize)]
pub struct Esmfold {
    pub model: Model,
    pub ports: Ports,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Esmfold {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub const PROGRAMS: [(Prog, &str); 3] = [(Prog::Lm, "lm"), (Prog::Trunk, "trunk"), (Prog::Sm, "sm")];

pub struct Program {
    pub bundle: Bundle<Esmfold>,
    pub loaded: Loaded,
    pub ranks: Ranks,
}

pub struct Device {
    pub lm: Program,
    pub trunk: Program,
    pub sm: Program,
}

impl Device {
    fn at(&self, prog: Prog) -> &Program {
        match prog {
            Prog::Lm => &self.lm,
            Prog::Trunk => &self.trunk,
            Prog::Sm => &self.sm,
        }
    }
}

impl Device {
    /// Program `prog`'s stream waits for the other two's queued work: the fold one queue, as esmf/demo's host ran it on a single
    /// stream (a launch never overlapping another program's work on the same GPU).
    fn serial(&self, prog: Prog) -> Result<(), String> {
        let p = self.at(prog);
        [&self.lm, &self.trunk, &self.sm].iter().try_for_each(|o| p.loaded.after(&o.loaded))
    }
}

impl fold::Step for Device {
    fn write(&mut self, prog: Prog, name: &str, bytes: &[u8]) -> Result<(), String> {
        let p = self.at(prog);
        let port = p.bundle.reference.served.ports.input(name, bytes.len())?;
        p.loaded.write(port.home, port.at.start, bytes)
    }

    fn zero(&mut self, prog: Prog, name: &str, len: usize) -> Result<(), String> {
        self.serial(prog)?;
        let p = self.at(prog);
        let port = p.bundle.reference.served.ports.input(name, len)?;
        p.loaded.zero(port.home, port.at.start, len)
    }

    fn copy(&mut self, from: (Prog, &str), to: (Prog, &str), len: usize) -> Result<(), String> {
        self.serial(to.0)?;
        let (src, dst) = (self.at(from.0), self.at(to.0));
        let out = src.bundle.reference.served.ports.output(from.1, len)?;
        let into = dst.bundle.reference.served.ports.input(to.1, len)?;
        dst.loaded.queued((into.home, into.at.start), &src.loaded, (out.home, out.at.start), len)?;
        synced(&dst.loaded, || format!("copy {:?}.{} -> {:?}.{} ({len} B)", from.0, from.1, to.0, to.1))
    }

    fn launch(&mut self, prog: Prog) -> Result<(), String> {
        self.serial(prog)?;
        let p = self.at(prog);
        p.loaded.launch(&p.bundle.reference.kernel)?;
        synced(&p.loaded, || format!("launch {prog:?}"))
    }

    fn read(&mut self, prog: Prog, name: &str, len: usize) -> Result<Vec<u8>, String> {
        let p = self.at(prog);
        let port = p.bundle.reference.served.ports.output(name, len)?;
        p.loaded.finish()?;
        p.loaded.read(port.home, port.at.start, len)
    }
}

/// `HS_ESMFOLD_SYNC`: each launch and copy awaited, a fault named by the step that made it.
fn synced(l: &Loaded, what: impl Fn() -> String) -> Result<(), String> {
    match std::env::var_os("HS_ESMFOLD_SYNC") {
        Some(_) => l.finish().map_err(|e| format!("{}: {e}", what())),
        None => Ok(()),
    }
}

pub fn wanted(prog: Prog, l: usize) -> Vec<(&'static str, bool, usize)> {
    use fold::{LDDT_ATOMS, LDDT_BINS, PAIR, SINGLE, SM_PAIR, SM_SINGLE};
    let pairs = l * l;
    match prog {
        Prog::Lm => vec![("ids", true, 4 * (l + 2)), ("aa", true, 4 * l), ("s0", false, 4 * l * SINGLE)],
        Prog::Trunk => vec![
            ("s0", true, 4 * l * SINGLE),
            ("s_prev", true, 2 * l * SINGLE),
            ("z_prev", true, 2 * pairs * PAIR),
            ("bins", true, 4 * pairs),
            ("s", false, 2 * l * SINGLE),
            ("z", false, 2 * pairs * PAIR),
            ("sm_s", false, 2 * l * SM_SINGLE),
            ("sm_z", false, 2 * pairs * SM_PAIR),
        ],
        Prog::Sm => vec![
            ("sm_s", true, 2 * l * SM_SINGLE),
            ("sm_z", true, 2 * pairs * SM_PAIR),
            ("backbone", true, 4 * l * 9),
            ("frame", false, 4 * l * 7),
            ("lddt", false, 4 * l * LDDT_ATOMS * LDDT_BINS),
            ("bins", false, 4 * pairs),
        ],
    }
}

pub fn check(prog: Prog, ports: &Ports, l: usize) -> Result<(), String> {
    for (name, input, bytes) in wanted(prog, l) {
        let p = if input { ports.input(name, bytes)? } else { ports.output(name, bytes)? };
        if p.at.end - p.at.start != bytes as u64 {
            return Err(format!(
                "esmfold {prog:?}: port {name} holds {} B, not {bytes} ({l} residues)",
                p.at.end - p.at.start
            ));
        }
    }
    Ok(())
}
