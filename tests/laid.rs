use std::path::Path;

use hyperserve::bundle::{Bundle, Fill, Laid, Served, Transform};
use hyperserve::weights::Checkpoint;
use serde::Deserialize;

#[derive(Deserialize)]
struct Any {
    model: Model,
}

#[derive(Deserialize)]
struct Model {
    ranks: usize,
}

impl Served for Any {
    fn ranks(&self) -> usize {
        self.model.ranks
    }
}

#[test]
#[ignore = "compares against hk's laid weights: HS_LAID_BUNDLE=DIR HS_LAID_WEIGHTS=DIR HS_LAID_CK=DIR"]
fn every_weight_hyperserve_lays_is_byte_for_byte_the_one_hk_lays() {
    let env = |k: &str| std::env::var_os(k).unwrap_or_else(|| panic!("{k}"));
    let (bundle, hk, ck) = (env("HS_LAID_BUNDLE"), env("HS_LAID_WEIGHTS"), env("HS_LAID_CK"));
    let b = Bundle::<Any>::read(Path::new(&bundle), 0).unwrap_or_else(|e| panic!("{e}"));
    let ck = Checkpoint::open(Path::new(&ck)).unwrap_or_else(|e| panic!("{e}"));
    let hk = Checkpoint::open(Path::new(&hk)).unwrap_or_else(|e| panic!("{e}"));
    let (mut same, mut differ) = (0, Vec::new());
    for i in &b.inputs.inputs {
        let Fill::Checkpoint { transform, tensors, cut, laid } = &i.fill else { continue };
        let ours = ck.laid(transform, tensors, cut.as_ref(), laid).unwrap_or_else(|e| panic!("{}: {e}", i.name));
        let theirs = hk.laid(&Transform::Stack, std::slice::from_ref(&i.name), None, &Laid::default()).unwrap_or_else(|e| panic!("hk's {}: {e}", i.name));
        match ours.iter().zip(&theirs).position(|(a, b)| a != b) {
            None if ours.len() == theirs.len() => same += 1,
            at => differ.push(format!("{} ({:?}): {} vs {} B, first differing byte {at:?}", i.name, transform, ours.len(), theirs.len())),
        }
    }
    println!("LAID same={same} differ={}", differ.len());
    assert!(differ.is_empty(), "{} of {} weights differ:\n{}", differ.len(), same + differ.len(), differ.join("\n"));
    assert!(same > 0, "no checkpoint weights in the bundle");
}
