use std::path::{Path, PathBuf};

use clap::Args;

use crate::bundle::Bundle;
use crate::esmfold::fold;
use crate::esmfold::{Device, Esmfold, PROGRAMS, Program, check};
use crate::gpu::{cuda, load, ranks};
use crate::weights::Checkpoint;

pub const GPUS: usize = 1;

#[derive(Args)]
pub struct Flags {
    #[arg(
        long,
        help = "The protein: one-letter amino acids",
        conflicts_with = "fasta",
        required_unless_present = "fasta"
    )]
    sequence: Option<String>,
    #[arg(long, help = "A FASTA (or plain) file whose first record is the protein")]
    fasta: Option<PathBuf>,
    #[arg(long, default_value_t = 3, help = "Timed folds, after one warm-up fold")]
    folds: usize,
    #[arg(
        long,
        help = "A file to write the last fold's backbone (N, CA, C) to, as PDB with CA pLDDT in the B-factor"
    )]
    out: Option<PathBuf>,
}

pub fn bundle(_: &Flags) -> &'static str {
    "esmfold"
}

pub fn lengths(root: &Path) -> Vec<usize> {
    let mut ls: Vec<usize> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .filter(|l: &usize| PROGRAMS.iter().all(|(_, d)| root.join(l.to_string()).join(d).is_dir()))
        .collect();
    ls.sort();
    ls
}

pub fn run(kernels: &Path, weights: &Path, flags: Flags) -> Result<(), String> {
    let text = match (&flags.sequence, &flags.fasta) {
        (Some(s), _) => s.clone(),
        (None, Some(p)) => std::fs::read_to_string(p).map_err(|e| format!("--fasta {}: {e}", p.display()))?,
        (None, None) => return Err("--sequence AA or --fasta FILE".into()),
    };
    let aa = fold::sequence(&text)?;
    let l = aa.len();
    let dir = kernels.join(l.to_string());
    if !dir.is_dir() {
        return Err(format!(
            "no esmfold bundle for {l} residues under {}: the kernels are compiled per length; lengths present: {:?}",
            kernels.display(),
            lengths(kernels)
        ));
    }
    cuda::visible(GPUS)?;
    let ck = Checkpoint::open(weights)?;
    let mut programs = Vec::with_capacity(PROGRAMS.len());
    for (prog, sub) in PROGRAMS {
        let bundle = Bundle::<Esmfold>::read(&dir.join(sub), 0)?;
        check(prog, &bundle.reference.served.ports, l)?;
        let ranks = ranks::Ranks::join(0, bundle.reference.kernel.symmetric)?;
        let loaded = load::load(&bundle, &ranks, Some(&ck), &[])?;
        programs.push(Program { bundle, loaded, ranks });
    }
    let [lm, trunk, sm]: [Program; 3] = programs.try_into().map_err(|_| "three esmfold programs")?;
    let mut dev = Device { lm, trunk, sm };
    if std::env::var_os("HS_ESMFOLD_DEBUG").is_some() {
        print!("{}", fold::probe(&mut dev, &aa)?);
    }
    let r = fold::run(&mut dev, &aa, flags.folds)?;
    print!("{}", r.lines);
    if let Some(out) = flags.out {
        std::fs::write(&out, &r.pdb).map_err(|e| format!("--out {}: {e}", out.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_are_the_folders_holding_all_three_programs() {
        let root = std::env::temp_dir().join(format!("hyperserve-esmfold-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (l, subs) in [
            ("164", &["lm", "trunk", "sm"][..]),
            ("76", &["lm", "trunk", "sm"]),
            ("200", &["lm", "trunk"]),
            ("x", &["lm", "trunk", "sm"]),
        ] {
            subs.iter().for_each(|s| std::fs::create_dir_all(root.join(l).join(s)).unwrap());
        }
        assert_eq!(lengths(&root), [76, 164]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
