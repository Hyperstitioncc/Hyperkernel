use std::path::PathBuf;

use hyperserve::bundle::{Blob, Bundle, Dtype, Fill, Record};
use hyperserve::openpilot::Openpilot;
use hyperserve::openpilot::drive::{SLICES, reads};
use hyperserve::weights::Checkpoint;

const FREE: u64 = 275_921_829_888;
const SLACK: u64 = 4 << 30;

fn inside(blob: &Blob, size: u64) -> bool {
    blob.at.checked_add(blob.len).is_some_and(|end| end <= size)
}

#[test]
fn the_openpilot_bundle_boots_at_a_b300_and_holds_the_drives_ports() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/openpilot");
    let b = Bundle::<Openpilot>::read(&dir, 0).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let (r, size) = (&b.reference, b.blobs.len() as u64);
    assert_eq!(r.served.model.name, "openpilot");
    assert_eq!(r.served.model.ranks, 1);
    assert!(r.kernel.workspace + SLACK < FREE, "a {} B arena", r.kernel.workspace);
    assert!(r.homes.iter().all(|h| h.at.start <= h.at.end));
    for x in &r.image {
        if let Record::Write { blob, .. } = x {
            assert!(inside(blob, size), "a write's blob {blob:?} past blobs.bin's {size} B");
        }
    }
    for i in &b.inputs.inputs {
        let home = &r.homes[i.home];
        assert!(i.at.end <= home.at.end - home.at.start, "input {} past its home", i.name);
        match &i.fill {
            Fill::Blob { blob } => assert!(inside(blob, size), "input {}'s blob past blobs.bin", i.name),
            Fill::Checkpoint { .. } | Fill::Zeros => {}
            other => panic!("input {}: {other:?}, not a checkpoint, blob or zeros", i.name),
        }
    }
    assert!(
        b.inputs.inputs.iter().filter(|i| matches!(i.fill, Fill::Checkpoint { .. })).count() > 200,
        "the weights the checkpoint's"
    );
    assert!(size < 1 << 20, "blobs.bin of {size} B: no weights in the bundle");
    for cubin in r.kernel.loaders.iter().map(|(c, _)| c.as_str()).chain(["entry.cubin"]) {
        assert!(b.cubin(cubin).is_ok_and(|c| !c.is_empty()), "no {cubin}");
    }
    let ports = &r.served.ports;
    for (name, bytes) in
        [("img", 128 * 256 * 24 * 2), ("feats", 8 * 512 * 2), ("desire", 200 * 2), ("traffic", 8 * 2)]
    {
        let p = ports.input(name, bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.dtype, p.at.end - p.at.start), (Dtype::Bf16, bytes as u64), "input {name}");
        assert!(b.inputs.inputs.iter().any(|i| i.name == name && i.home == p.home), "input {name} unbound");
    }
    for (name, bytes) in reads() {
        let p = ports.output(name, bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.dtype, if name == "hidden_state" { Dtype::Bf16 } else { Dtype::F32 }, "output {name}");
    }
    assert_eq!(reads().len(), SLICES.len() - 1);
}

#[test]
#[ignore = "reads the converted checkpoint: HS_OPENPILOT_WEIGHTS=DIR"]
fn every_weight_of_the_bundle_is_a_tensor_of_its_bytes() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/openpilot");
    let b = Bundle::<Openpilot>::read(&dir, 0).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let weights = std::env::var_os("HS_OPENPILOT_WEIGHTS")
        .expect("HS_OPENPILOT_WEIGHTS: the converted checkpoint's directory");
    let ck = Checkpoint::open(std::path::Path::new(&weights)).unwrap_or_else(|e| panic!("{e}"));
    for i in &b.inputs.inputs {
        if let Fill::Checkpoint { transform, tensors, cut, laid } = &i.fill {
            let bytes = ck.laid(transform, tensors, cut.as_ref(), laid).unwrap_or_else(|e| panic!("{}: {e}", i.name));
            assert_eq!(bytes.len() as u64, i.at.end - i.at.start, "{}", i.name);
        }
    }
}
