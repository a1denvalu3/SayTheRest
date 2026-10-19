//! Pinned native runtime/model assets. Downloading them never installs Python.
use anyhow::{Context, Result, bail, ensure};
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub const VERSION: &str = "1.24.2";

#[derive(Clone, Debug, Deserialize)]
pub struct Asset {
    pub path: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Deserialize)]
pub struct ModelAssets {
    pub repository: String,
    pub revision: String,
    pub upstream_revision: String,
    pub files: Vec<Asset>,
}

pub fn model_assets() -> ModelAssets {
    serde_json::from_str(include_str!("../data/qwen-onnx-assets.json"))
        .expect("compiled native model manifest")
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn download(
    asset: &Asset,
    destination: &Path,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<()> {
    ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
    if destination.is_file()
        && fs::metadata(destination)?.len() == asset.size
        && sha256_file(destination)? == asset.sha256
    {
        progress(asset.size);
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let pending = destination.with_extension("download");
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()?;
        let mut response = client.get(&asset.url).send()?.error_for_status()?;
        let mut file = File::create(&pending)?;
        let mut hasher = Sha256::new();
        let mut received = 0;
        let mut buffer = [0; 128 * 1024];
        loop {
            ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
            let count = response.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            received += count as u64;
            ensure!(
                received <= asset.size,
                "native asset exceeds its pinned size"
            );
            file.write_all(&buffer[..count])?;
            hasher.update(&buffer[..count]);
            progress(received);
        }
        ensure!(
            received == asset.size && format!("{:x}", hasher.finalize()) == asset.sha256,
            "native asset checksum mismatch: {}",
            asset.path
        );
        file.sync_all()?;
        drop(file);
        fs::rename(&pending, destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(pending);
    }
    result
}

#[derive(Clone)]
pub struct NativeRuntimeManager {
    root: PathBuf,
}

impl NativeRuntimeManager {
    pub fn new(state_dir: &Path) -> Self {
        Self {
            root: state_dir.join("runtimes/onnx"),
        }
    }

    pub fn install(&self, cuda: bool) -> Result<PathBuf> {
        if !cuda {
            let packaged = std::env::current_exe()?
                .parent()
                .context("missing application directory")?
                .join("runtime/native");
            let library = runtime_library(&packaged);
            if library.is_file() {
                return Ok(library);
            }
        }
        let (asset, digest, size) = runtime_asset(cuda)?;
        fs::create_dir_all(&self.root)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join("install.lock"))?;
        lock.lock()?;
        let destination = self
            .root
            .join(format!("{VERSION}-{}", if cuda { "cuda" } else { "cpu" }));
        let library = runtime_library(&destination);
        if library.is_file()
            && fs::read_to_string(destination.join("installed"))
                .is_ok_and(|installed| installed == digest)
        {
            if cuda {
                self.install_cuda_libraries(&destination)?;
            }
            return Ok(library);
        }
        let archive = self.root.join(asset);
        let pinned = Asset {
            path: asset.into(),
            url: format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/{asset}"
            ),
            sha256: digest.into(),
            size,
        };
        eprintln!(
            "Preparing native ONNX Runtime {VERSION} ({})",
            if cuda { "CUDA" } else { "CPU" }
        );
        download(&pinned, &archive, &AtomicBool::new(false), |_| {})?;
        let staging = self.root.join(".installing");
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;
        if asset.ends_with(".zip") {
            zip::ZipArchive::new(File::open(&archive)?)?.extract(&staging)?;
        } else {
            tar::Archive::new(GzDecoder::new(File::open(&archive)?)).unpack(&staging)?;
        }
        let source = fs::read_dir(&staging)?
            .next()
            .context("empty native runtime archive")??
            .path();
        ensure!(
            runtime_library(&source).is_file(),
            "runtime archive is missing its library"
        );
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(source, &destination)?;
        fs::remove_dir_all(staging)?;
        fs::write(destination.join("installed"), digest)?;
        let _ = fs::remove_file(archive);
        if cuda {
            self.install_cuda_libraries(&destination)?;
        }
        Ok(library)
    }

    fn install_cuda_libraries(&self, destination: &Path) -> Result<()> {
        let manifest = include_str!("../data/onnx-cuda-assets.json");
        let version = format!("{:x}", Sha256::digest(manifest.as_bytes()));
        let receipt = destination.join("cuda-libraries");
        if fs::read_to_string(&receipt).is_ok_and(|installed| installed == version) {
            return Ok(());
        }
        let platforms: std::collections::HashMap<String, Vec<Asset>> =
            serde_json::from_str(manifest)?;
        let assets = platforms
            .get(std::env::consts::OS)
            .context("unsupported native CUDA platform")?;
        for asset in assets {
            let archive = self.root.join(&asset.path);
            download(asset, &archive, &AtomicBool::new(false), |_| {})?;
            let mut zip = zip::ZipArchive::new(File::open(&archive)?)?;
            for index in 0..zip.len() {
                let mut file = zip.by_index(index)?;
                if file.is_dir() {
                    continue;
                }
                let path = Path::new(file.name());
                let name = path
                    .file_name()
                    .context("invalid CUDA library filename")?
                    .to_owned();
                let native = name.to_string_lossy();
                let target = if native.contains(".so") || native.ends_with(".dll") {
                    destination.join("lib").join(name)
                } else if native.to_ascii_lowercase().contains("license") {
                    destination.join("licenses").join(&asset.path).join(name)
                } else {
                    continue;
                };
                fs::create_dir_all(target.parent().unwrap())?;
                let mut output = File::create(target)?;
                std::io::copy(&mut file, &mut output)?;
            }
            drop(zip);
            let _ = fs::remove_file(archive);
        }
        fs::write(receipt, version)?;
        Ok(())
    }
}

pub fn runtime_library(root: &Path) -> PathBuf {
    root.join("lib").join(if cfg!(windows) {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so.1.24.2"
    })
}

fn runtime_asset(cuda: bool) -> Result<(&'static str, &'static str, u64)> {
    match (std::env::consts::OS, std::env::consts::ARCH, cuda) {
        ("linux", "x86_64", false) => Ok((
            "onnxruntime-linux-x64-1.24.2.tgz",
            "43725474ba5663642e17684717946693850e2005efbd724ac72da278fead25e6",
            8123282,
        )),
        ("linux", "x86_64", true) => Ok((
            "onnxruntime-linux-x64-gpu-1.24.2.tgz",
            "bcb42da041f42192e5579de175f7410313c114740a611e230afe9d79be65cc49",
            205369881,
        )),
        ("windows", "x86_64", false) => Ok((
            "onnxruntime-win-x64-1.24.2.zip",
            "8e3e9c826375352e29cb2614fe44f3d7a4b0ff7b8028ad7a456af9d949a7e8b0",
            74075355,
        )),
        ("windows", "x86_64", true) => Ok((
            "onnxruntime-win-x64-gpu-1.24.2.zip",
            "4152b67ea67405cf642fdc484a4e24fb1d9bfda7a78dfd5f756c73812a40bdcd",
            280855316,
        )),
        _ => bail!("native Qwen currently supports Linux/Windows x86-64"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiled_model_assets_have_immutable_sources_and_cover_each_precision() {
        let manifest = model_assets();
        let mut paths = std::collections::HashSet::new();
        for asset in &manifest.files {
            assert!(paths.insert(&asset.path));
            assert!(asset.size > 0);
            assert_eq!(asset.sha256.len(), 64);
            assert!(
                asset.url.contains(&manifest.revision)
                    || asset.url.contains(&manifest.upstream_revision)
            );
            assert!(!Path::new(&asset.path).is_absolute());
            assert!(!asset.path.contains(".."));
        }
        for precision in ["fp32", "int4"] {
            assert!(paths.contains(&format!("{precision}/talker_cache.onnx")));
            assert!(paths.contains(&format!("{precision}/speaker_encoder.onnx")));
            assert!(paths.contains(&format!("{precision}/tok_encoder.onnx")));
            assert!(paths.contains(&format!("{precision}/tok_decoder.onnx")));
        }
    }
    #[test]
    fn cached_asset_hash_is_checked_before_reuse() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("asset");
        fs::write(&path, b"model").unwrap();
        let asset = Asset {
            path: "asset".into(),
            url: "http://127.0.0.1:1/unreachable".into(),
            sha256: format!("{:x}", Sha256::digest(b"model")),
            size: 5,
        };
        download(&asset, &path, &AtomicBool::new(false), |bytes| {
            assert_eq!(bytes, 5)
        })
        .unwrap();
        fs::write(&path, b"wrong").unwrap();
        assert!(download(&asset, &path, &AtomicBool::new(false), |_| {}).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"wrong");
    }
}
