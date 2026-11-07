use std::path::PathBuf;

use hyperserve::bundle::{Bundle, Fill};
use hyperserve::weights::Checkpoint;
use hyperserve::esmfold::cli::lengths;
use hyperserve::esmfold::{Esmfold, PROGRAMS, check};

const FREE: u64 = 275_921_829_888;

#[test]
#[ignore = "reads kernels/b300/esmfold/<residues>/{lm,trunk,sm}, exported by hk export from esmf/demo"]
fn every_esmfold_length_holds_three_programs_whose_ports_fit_a_fold() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/esmfold");
    let ls = lengths(&root);
    assert!(!ls.is_empty(), "no lengths under {}", root.display());
    for l in ls {
        let mut arena = 0;
        for (prog, sub) in PROGRAMS {
            let dir = root.join(l.to_string()).join(sub);
            let b = Bundle::<Esmfold>::read(&dir, 0).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            assert_eq!(b.reference.served.model.ranks, 1, "{}", dir.display());
            check(prog, &b.reference.served.ports, l).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            assert!(b.cubin("entry.cubin").is_ok_and(|c| !c.is_empty()), "{}: no entry.cubin", dir.display());
            arena += b.reference.kernel.workspace;
        }
        assert!(arena < FREE, "{l} residues: three arenas of {arena} B on one GPU");
    }
}

#[test]
#[ignore = "reads the converted checkpoint: HS_ESMFOLD_WEIGHTS=DIR (scripts/esmfold_safetensors.py's output)"]
fn every_weight_of_every_esmfold_bundle_is_laid_from_the_checkpoint_at_its_bytes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/b300/esmfold");
    let weights = std::env::var_os("HS_ESMFOLD_WEIGHTS").expect("HS_ESMFOLD_WEIGHTS: the converted checkpoint's directory");
    let ck = Checkpoint::open(std::path::Path::new(&weights)).unwrap_or_else(|e| panic!("{e}"));
    let ls = lengths(&root);
    assert!(!ls.is_empty(), "no lengths under {}", root.display());
    for l in ls {
        for (_, sub) in PROGRAMS {
            let dir = root.join(l.to_string()).join(sub);
            let b = Bundle::<Esmfold>::read(&dir, 0).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            for i in &b.inputs.inputs {
                if let Fill::Checkpoint { transform, tensors, cut, laid } = &i.fill {
                    let bytes = ck.laid(transform, tensors, cut.as_ref(), laid).unwrap_or_else(|e| panic!("{} {}: {e}", dir.display(), i.name));
                    assert_eq!(bytes.len() as u64, i.at.end - i.at.start, "{} {}", dir.display(), i.name);
                }
            }
        }
    }
}
