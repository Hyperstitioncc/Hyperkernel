pub mod cli;
pub mod drive;

use serde::Deserialize;

pub use crate::bundle::{Port, Ports};

use crate::bundle::{Bundle, Served};
use crate::gpu::load::Loaded;

#[derive(Deserialize)]
pub struct Openpilot {
    pub model: Model,
    pub ports: Ports,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Openpilot {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub struct Device<'a> {
    pub bundle: &'a Bundle<Openpilot>,
    pub loaded: Loaded,
}

impl drive::Step for Device<'_> {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
        let ports = &self.bundle.reference.served.ports;
        items.iter().try_for_each(|&(name, bytes)| ports.input(name, bytes.len()).and_then(|p| self.loaded.write(p.home, p.at.start, bytes)))
    }

    fn launch(&mut self) -> Result<(), String> {
        self.loaded.launch(&self.bundle.reference.kernel)?;
        self.loaded.finish()
    }

    fn read_all(&self, items: &[(&str, usize)]) -> Result<Vec<Vec<u8>>, String> {
        let ports = &self.bundle.reference.served.ports;
        items.iter().map(|&(name, bytes)| ports.output(name, bytes).and_then(|p| self.loaded.read(p.home, p.at.start, bytes))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_facts_read_beside_the_kernel_and_ports_check_their_bytes() {
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "program=openpilot",
            "model": { "name": "openpilot", "ranks": 1 },
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena"], "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false } },
            "homes": [], "image": [],
            "ports": {
                "inputs": [{ "name": "img", "home": 3, "at": { "start": 0, "end": 64 }, "dtype": "bf16", "shape": [4, 8] }],
                "outputs": [{ "name": "plan", "home": 5, "at": { "start": 16, "end": 80 }, "dtype": "f32", "shape": [1, 16] }]
            }
        });
        let r: crate::bundle::Reference<Openpilot> = serde_json::from_value(json).expect("a reference");
        let ports = &r.served.ports;
        assert_eq!(r.served.ranks(), 1);
        assert_eq!(r.kernel.globals.rings, "");
        assert_eq!(ports.input("img", 64).unwrap().home, 3);
        assert!(ports.input("img", 65).is_err());
        assert_eq!(ports.output("plan", 40).unwrap().at.start, 16);
        assert!(ports.output("img", 4).is_err());
    }
}
