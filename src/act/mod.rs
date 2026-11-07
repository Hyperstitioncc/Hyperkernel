pub mod cli;
pub mod tick;

use serde::Deserialize;

pub use crate::bundle::{Port, Ports};

use crate::bundle::{Bundle, Served};
use crate::gpu::Pinned;
use crate::gpu::load::Loaded;

#[derive(Deserialize)]
pub struct Act {
    pub model: Model,
    pub ports: Ports,
}

#[derive(Deserialize)]
pub struct Model {
    pub name: String,
    pub ranks: usize,
}

impl Served for Act {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

pub struct Device<'a> {
    pub bundle: &'a Bundle<Act>,
    pub loaded: Loaded,
    pub stage: Pinned,
}

impl tick::Step for Device<'_> {
    fn write_all(&mut self, items: &[(&str, &[u8])]) -> Result<(), String> {
        let ports = &self.bundle.reference.served.ports;
        items.iter().try_for_each(|&(name, bytes)| {
            ports.input(name, bytes.len()).and_then(|p| self.loaded.write_via(p.home, p.at.start, bytes, &self.stage))
        })
    }

    fn launch(&mut self) -> Result<(), String> {
        self.loaded.launch(&self.bundle.reference.kernel)?;
        self.loaded.finish()
    }

    fn read(&self, name: &str, bytes: usize) -> Result<Vec<u8>, String> {
        let p = self.bundle.reference.served.ports.output(name, bytes)?;
        self.loaded.read(p.home, p.at.start, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_facts_read_beside_the_kernel_and_ports_check_their_bytes() {
        let json = serde_json::json!({
            "format": "hyperserve-bundle-2", "point": "program=act",
            "model": { "name": "act", "ranks": 1 },
            "kernel": { "entries": [["k", 148]], "threads": 256, "cluster": 1, "smem": 0, "static_smem": 0, "cooperative": true, "stack": 0, "workspace": 0, "symmetric": 0,
                "l2": null, "homew": null, "params": ["arena"], "next": null, "loaders": [],
                "globals": { "fault": "f", "fabric": "g", "peers": "p", "peer_bytes": 32, "multicast": false } },
            "homes": [], "image": [],
            "ports": {
                "inputs": [
                    { "name": "image", "home": 2, "at": { "start": 0, "end": tick::IMAGE_BYTES }, "dtype": "u8", "shape": [307200, 3] },
                    { "name": "state", "home": 4, "at": { "start": 0, "end": tick::STATE_BYTES }, "dtype": "f32", "shape": [1, 16] }
                ],
                "outputs": [{ "name": "actions", "home": 7, "at": { "start": 0, "end": tick::ACTIONS_BYTES }, "dtype": "f32", "shape": [100, 14] }]
            }
        });
        let r: crate::bundle::Reference<Act> = serde_json::from_value(json).expect("a reference");
        let ports = &r.served.ports;
        assert_eq!(r.served.ranks(), 1);
        assert_eq!(ports.input("image", tick::IMAGE_BYTES).unwrap().home, 2);
        assert!(ports.input("state", tick::STATE_BYTES + 1).is_err());
        assert_eq!(ports.output("actions", tick::ACTIONS_BYTES).unwrap().home, 7);
    }
}
