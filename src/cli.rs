use std::path::{Path, PathBuf};

use clap::Args;

const KERNELS: &str = "kernels/b300";

#[derive(Args)]
pub struct At {
    #[arg(long, help = "The model's checkpoint directory of .safetensors files")]
    pub weights: PathBuf,
    #[arg(long, help = "A kernel bundle directory, in place of the one under kernels/b300 found above the executable or the working directory")]
    pub kernels: Option<PathBuf>,
}

impl At {
    pub fn ready(&self, bundle: &str) -> Result<PathBuf, String> {
        self.kernels(bundle, &starts())
    }

    pub fn kernels(&self, bundle: &str, starts: &[PathBuf]) -> Result<PathBuf, String> {
        match &self.kernels {
            Some(dir) => Ok(dir.clone()),
            None => find(bundle, starts),
        }
    }
}

pub fn find(bundle: &str, starts: &[PathBuf]) -> Result<PathBuf, String> {
    let root = starts.iter().flat_map(|s| s.ancestors()).map(|a| a.join(KERNELS)).find(|d| d.is_dir());
    root.map(|r| r.join(bundle)).ok_or_else(|| format!("bundle {bundle}: no {KERNELS} above {}", shown(starts)))
}

fn starts() -> Vec<PathBuf> {
    let exe = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
    exe.into_iter().chain(std::env::current_dir().ok()).collect()
}

fn shown(starts: &[PathBuf]) -> String {
    starts.iter().map(|s| s.display().to_string()).collect::<Vec<_>>().join(" or ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(tag: &str, bundles: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("hyperserve-cli-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        bundles.iter().for_each(|b| std::fs::create_dir_all(root.join(KERNELS).join(b)).unwrap());
        std::fs::create_dir_all(root.join("target/release")).unwrap();
        root
    }

    fn at(kernels: Option<&str>) -> At {
        At { weights: PathBuf::from("w"), kernels: kernels.map(PathBuf::from) }
    }

    #[test]
    fn a_bundle_is_found_under_the_first_kernels_folder_above_the_exe_then_the_cwd() {
        let (exe, cwd) = (tree("exe", &["c5"]), tree("cwd", &["c5"]));
        let found = |starts: &[PathBuf]| find("c5", starts).unwrap();
        assert_eq!(found(&[exe.join("target/release"), cwd.clone()]), exe.join("kernels/b300/c5"));
        assert_eq!(found(&[PathBuf::from("/"), cwd.join("target")]), cwd.join("kernels/b300/c5"));
        assert!(find("c5", &[PathBuf::from("/proc/self")]).unwrap_err().starts_with("bundle c5: no kernels/b300 above /proc/self"));
        [exe, cwd].iter().for_each(|r| std::fs::remove_dir_all(r).unwrap());
    }

    #[test]
    fn kernels_override_the_bundle() {
        let none: [PathBuf; 0] = [];
        assert_eq!(at(Some("/k")).kernels("c5", &none).unwrap(), PathBuf::from("/k"));
        assert!(at(None).kernels("c5", &none).is_err());
    }
}
