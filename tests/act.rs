use std::path::PathBuf;

use hyperserve::act::Act;
use hyperserve::act::tick::{ACTIONS_BYTES, IMAGE_BYTES, STATE_BYTES};
use hyperserve::bundle::{Blob, Bundle, Dtype, Fill, Record};
use hyperserve::weights::Checkpoint;

const FREE: u64 = 275_921_829_888;
const SLACK: u64 = 4 << 30;

fn inside(blob: &Blob, size: u64) -> bool {
    blob.at.checked_add(blob.len).is_some_and(|end| end <= size)
}

#[test]
#[ignore = "reads kernels/b300/act, exported by hk export from act/demo"]
fn the_act_bundle_boots_at_a_b300_and_holds_the_ticks_ports() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/act");
    let b = Bundle::<Act>::read(&dir, 0).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let (r, size) = (&b.reference, b.blobs.len() as u64);
    assert_eq!((r.served.model.name.as_str(), r.served.model.ranks), ("act", 1));
    assert!(r.kernel.workspace + SLACK < FREE, "a {} B arena", r.kernel.workspace);
    for x in &r.image {
        if let Record::Write { blob, .. } = x {
            assert!(inside(blob, size), "a write's blob {blob:?} past blobs.bin's {size} B");
        }
    }
    for i in &b.inputs.inputs {
        if let Fill::Blob { blob } = &i.fill {
            assert!(inside(blob, size), "input {}'s blob past blobs.bin", i.name);
        }
    }
    for cubin in r.kernel.loaders.iter().map(|(c, _)| c.as_str()).chain(["entry.cubin"]) {
        assert!(b.cubin(cubin).is_ok_and(|c| !c.is_empty()), "no {cubin}");
    }
    let ports = &r.served.ports;
    for (name, bytes, dtype) in [("image", IMAGE_BYTES, Dtype::U8), ("state", STATE_BYTES, Dtype::F32)] {
        let p = ports.input(name, bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.dtype, p.at.end - p.at.start), (dtype, bytes as u64), "input {name}");
    }
    let a = ports.output("actions", ACTIONS_BYTES).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((a.dtype, a.at.end - a.at.start), (Dtype::F32, ACTIONS_BYTES as u64));
}

#[test]
#[ignore = "reads the converted checkpoint: HS_ACT_WEIGHTS=DIR (scripts/act_safetensors.py's output)"]
fn every_weight_of_the_act_bundle_is_laid_from_the_checkpoint_at_its_bytes() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/act");
    let b = Bundle::<Act>::read(&dir, 0).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let weights = std::env::var_os("HS_ACT_WEIGHTS").expect("HS_ACT_WEIGHTS: the converted checkpoint's directory");
    let ck = Checkpoint::open(std::path::Path::new(&weights)).unwrap_or_else(|e| panic!("{e}"));
    let mut laid = 0;
    for i in &b.inputs.inputs {
        if let Fill::Checkpoint { transform, tensors, cut, laid: l } = &i.fill {
            let bytes = ck.laid(transform, tensors, cut.as_ref(), l).unwrap_or_else(|e| panic!("{}: {e}", i.name));
            assert_eq!(bytes.len() as u64, i.at.end - i.at.start, "{}", i.name);
            laid += 1;
        }
    }
    assert!(laid > 100, "{laid} weights laid");
}
