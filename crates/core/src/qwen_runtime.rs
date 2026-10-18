use crate::{CudaDevice, TorchFlavor};
use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

const UV_VERSION: &str = "0.12.3";
const PYTHON_VERSION: &str = "3.12.12";

#[derive(Clone, Debug)]
pub struct QwenRuntimeManager {
    root: PathBuf,
    manifest_dir: PathBuf,
}

impl QwenRuntimeManager {
    pub fn new(root: PathBuf, manifest_dir: PathBuf) -> Self {
        Self { root, manifest_dir }
    }

    pub fn python(&self) -> PathBuf {
        let preferred = self.python_for(Self::preferred_flavor());
        let cpu = self.python_for(TorchFlavor::Cpu);
        if preferred.is_file() || !cpu.is_file() {
            preferred
        } else {
            cpu
        }
    }

    pub fn preferred_flavor() -> TorchFlavor {
        CudaDevice::detect()
            .and_then(CudaDevice::torch_flavor)
            .unwrap_or(TorchFlavor::Cpu)
    }

    pub fn environment(&self, flavor: TorchFlavor) -> PathBuf {
        self.root.join(match flavor {
            TorchFlavor::Cpu => "venv",
            TorchFlavor::Cuda126 => "venv-cu126",
            TorchFlavor::Cuda128 => "venv-cu128",
            TorchFlavor::CudaLibraries => "venv-onnx-cu12",
        })
    }

    pub fn python_for(&self, flavor: TorchFlavor) -> PathBuf {
        if cfg!(windows) {
            self.environment(flavor).join("Scripts/python.exe")
        } else {
            self.environment(flavor).join("bin/python")
        }
    }

    pub fn owns_python(&self, python: &Path) -> bool {
        [TorchFlavor::Cpu, TorchFlavor::Cuda126, TorchFlavor::Cuda128]
            .iter()
            .any(|flavor| self.python_for(*flavor) == python)
    }

    pub fn worker(&self) -> PathBuf {
        self.manifest_dir.join("qwen_worker.py")
    }

    fn project_dir(&self) -> PathBuf {
        if self.manifest_dir.join("uv.lock").is_file() {
            self.manifest_dir.clone()
        } else {
            self.manifest_dir.join("qwen-runtime")
        }
    }

    pub fn is_installed(&self) -> bool {
        self.is_installed_for(Self::preferred_flavor())
    }

    fn marker(&self, flavor: TorchFlavor) -> PathBuf {
        self.root
            .join(format!("runtime-version-{}", flavor.extra()))
    }

    pub fn is_installed_for(&self, flavor: TorchFlavor) -> bool {
        self.python_for(flavor).is_file()
            && self.worker().is_file()
            && fs::read_to_string(self.marker(flavor))
                .is_ok_and(|value| value == self.runtime_version(flavor))
    }

    fn runtime_version(&self, flavor: TorchFlavor) -> String {
        let lock = fs::read(self.project_dir().join("uv.lock")).unwrap_or_default();
        format!(
            "uv={UV_VERSION}\npython={PYTHON_VERSION}\nqwen-tts=0.1.1\nextra={}\nlock={:x}\n",
            flavor.extra(),
            Sha256::digest(lock)
        )
    }

    pub fn install(&self) -> Result<()> {
        let preferred = Self::preferred_flavor();
        match self.install_for(preferred) {
            Err(error) if preferred != TorchFlavor::Cpu => {
                eprintln!("GPU runtime setup failed; using CPU: {error:#}");
                self.install_for(TorchFlavor::Cpu)
            }
            result => result,
        }
    }

    pub fn install_for(&self, flavor: TorchFlavor) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join("install.lock"))?;
        lock.lock()
            .context("failed to lock inference runtime installation")?;
        if self.is_installed_for(flavor) {
            return Ok(());
        }
        eprintln!("Preparing local inference runtime ({})", flavor.extra());
        let uv = self.install_uv()?;
        let python_install_dir = self.root.join("python");
        let status = Command::new(&uv)
            .args(["python", "install", PYTHON_VERSION, "--no-bin"])
            .env("UV_PYTHON_INSTALL_DIR", &python_install_dir)
            .status()
            .context("failed to start the managed Python installer")?;
        anyhow::ensure!(
            status.success(),
            "managed Python installation failed: {status}"
        );

        let project_dir = self.project_dir();
        for file in ["pyproject.toml", "uv.lock"] {
            anyhow::ensure!(
                project_dir.join(file).is_file(),
                "packaged Qwen runtime is missing {file}"
            );
        }
        anyhow::ensure!(
            self.worker().is_file(),
            "packaged Qwen runtime is missing qwen_worker.py"
        );
        let status = Command::new(&uv)
            .args([
                "sync",
                "--project",
                &project_dir.to_string_lossy(),
                "--locked",
                "--no-dev",
                "--extra",
                flavor.extra(),
                "--python",
                PYTHON_VERSION,
            ])
            .env("UV_PYTHON_INSTALL_DIR", &python_install_dir)
            .env("UV_PROJECT_ENVIRONMENT", self.environment(flavor))
            .status()
            .context("failed to start the pinned Qwen dependency installer")?;
        anyhow::ensure!(
            status.success(),
            "Qwen dependency installation failed: {status}"
        );
        anyhow::ensure!(
            self.python_for(flavor).is_file(),
            "managed Qwen Python was not created"
        );
        let marker = self.marker(flavor).with_extension("tmp");
        fs::write(&marker, self.runtime_version(flavor))?;
        sync_file(&marker)?;
        replace_file(&marker, &self.marker(flavor))?;
        Ok(())
    }

    fn install_uv(&self) -> Result<PathBuf> {
        let executable = if cfg!(windows) {
            self.root.join("uv/uv.exe")
        } else {
            self.root.join("uv/uv")
        };
        if executable.is_file() {
            return Ok(executable);
        }
        let (asset, sha256) = if cfg!(windows) {
            (
                "uv-x86_64-pc-windows-msvc.zip",
                "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e",
            )
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            (
                "uv-x86_64-unknown-linux-gnu.tar.gz",
                "600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101",
            )
        } else {
            bail!("the managed Qwen runtime currently supports Linux and Windows x86-64")
        };
        let archive = self.root.join(asset);
        download(
            &format!("https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}"),
            &archive,
            sha256,
        )?;
        let staging = self.root.join("uv.installing");
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;
        if cfg!(windows) {
            extract_zip(&archive, &staging)?;
        } else {
            let decoder = GzDecoder::new(File::open(&archive)?);
            let mut tar = tar::Archive::new(decoder);
            for entry in tar.entries()? {
                let mut entry = entry?;
                anyhow::ensure!(
                    entry.unpack_in(&staging)?,
                    "uv archive contains an unsafe path"
                );
            }
        }
        let found = find_named_file(&staging, if cfg!(windows) { "uv.exe" } else { "uv" })?;
        let destination = self.root.join("uv");
        fs::create_dir_all(&destination)?;
        fs::copy(found, &executable)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))?;
        }
        let _ = fs::remove_file(archive);
        let _ = fs::remove_dir_all(staging);
        Ok(executable)
    }
}

fn download(url: &str, destination: &Path, expected_sha256: &str) -> Result<()> {
    let mut response = reqwest::blocking::get(url)?.error_for_status()?;
    let mut output = File::create(destination)?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        anyhow::ensure!(
            total <= 100 * 1024 * 1024,
            "runtime download exceeds 100 MB"
        );
        output.write_all(&buffer[..count])?;
        digest.update(&buffer[..count]);
    }
    output.sync_all()?;
    let received = format!("{:x}", digest.finalize());
    anyhow::ensure!(
        received == expected_sha256,
        "runtime download checksum mismatch"
    );
    Ok(())
}

fn extract_zip(archive: &Path, destination: &Path) -> Result<()> {
    let mut archive = zip::ZipArchive::new(File::open(archive)?)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let relative = entry
            .enclosed_name()
            .context("uv ZIP contains an unsafe path")?;
        let output = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(output)?;
        } else {
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            std::io::copy(&mut entry, &mut File::create(output)?)?;
        }
    }
    Ok(())
}

fn find_named_file(root: &Path, name: &str) -> Result<PathBuf> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Ok(path) = find_named_file(&entry.path(), name) {
                return Ok(path);
            }
        } else if entry.file_name() == name {
            return Ok(entry.path());
        }
    }
    bail!("runtime archive does not contain {name}")
}

fn sync_file(path: &Path) -> Result<()> {
    fs::OpenOptions::new().write(true).open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_paths_never_require_a_system_python() {
        let root = tempfile::tempdir().unwrap();
        let manager =
            QwenRuntimeManager::new(root.path().join("managed"), root.path().join("bundle"));
        assert!(manager.python().starts_with(root.path()));
        assert!(!manager.is_installed());
    }

    #[test]
    fn finds_nested_runtime_binary() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("asset/bin");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("uv"), b"binary").unwrap();
        assert_eq!(
            find_named_file(root.path(), "uv").unwrap(),
            nested.join("uv")
        );
    }

    #[test]
    #[ignore = "downloads the pinned managed Python and Qwen inference environment"]
    fn installs_self_contained_qwen_environment() {
        let root = PathBuf::from(std::env::var_os("SAYIT_QWEN_RUNTIME_ROOT").unwrap());
        let manifest = PathBuf::from(std::env::var_os("SAYIT_QWEN_MANIFEST_DIR").unwrap());
        let manager = QwenRuntimeManager::new(root, manifest);
        manager.install_for(TorchFlavor::Cpu).unwrap();
        let output = Command::new(manager.python_for(TorchFlavor::Cpu))
            .args([
                "-c",
                "import qwen_tts, torch, torchaudio; print(torch.__version__, torchaudio.__version__)",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("2.9.1+cpu"));
    }
}
