//! Immutable Phonon assets and validation shared by installation, voice import,
//! and the isolated runtime. No Hub client is involved in synthesis.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path, sync::OnceLock};

pub const MAX_VOICE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize)]
pub struct Asset {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelAssets {
    pub id: String,
    pub repository: String,
    pub revision: String,
    pub files: Vec<Asset>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VoiceAsset {
    pub id: String,
    pub language: String,
    pub source_path: String,
    #[serde(flatten)]
    pub file: Asset,
}

#[derive(Debug, Deserialize)]
pub struct VoicePack {
    pub repository: String,
    pub revision: String,
    pub voices: Vec<VoiceAsset>,
}

pub fn model_assets() -> &'static ModelAssets {
    static MANIFEST: OnceLock<ModelAssets> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        serde_json::from_str(include_str!("../data/phonon-assets.json"))
            .expect("bundled Phonon manifest")
    })
}

pub fn voice_pack() -> &'static VoicePack {
    static MANIFEST: OnceLock<VoicePack> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        serde_json::from_str(include_str!("../data/phonon-voices.json"))
            .expect("bundled Phonon voices")
    })
}

/// Includes the voice pack, so a newly bundled pack offers a normal model
/// update while keeping custom profiles separate from managed model assets.
pub fn manifest_fingerprint() -> String {
    let mut hash = Sha256::new();
    hash.update(include_bytes!("../data/phonon-assets.json"));
    hash.update(include_bytes!("../data/phonon-voices.json"));
    format!("{:x}", hash.finalize())
}

pub fn read_voice(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((MAX_VOICE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    validate_voice(&bytes)?;
    Ok(bytes)
}

pub fn validate_voice(bytes: &[u8]) -> Result<()> {
    let invalid = "Choose a Phonon safetensors voice with finite, nonzero speaker latents, no larger than 8 MB";
    ensure!((9..=MAX_VOICE_BYTES).contains(&bytes.len()), "{invalid}");
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    ensure!((1..=65_536).contains(&header_length), "{invalid}");
    let start = 8 + header_length as usize;
    ensure!(start < bytes.len(), "{invalid}");
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..start]).context(invalid)?;
    let header = header.as_object().context(invalid)?;
    ensure!(
        header.contains_key("speaker_wavs")
            && header
                .keys()
                .all(|key| key == "speaker_wavs" || key == "__metadata__"),
        "{invalid}"
    );
    let tensor = &header["speaker_wavs"];
    let shape = tensor["shape"].as_array().context(invalid)?;
    ensure!(
        shape.len() == 3 && shape[0].as_u64() == Some(1) && shape[1].as_u64() == Some(512),
        "{invalid}"
    );
    let frames = shape[2].as_u64().context(invalid)?;
    ensure!((1..=4096).contains(&frames), "{invalid}");
    let dtype = tensor["dtype"].as_str().context(invalid)?;
    let width = match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        _ => anyhow::bail!(invalid),
    };
    let length = 512 * frames as usize * width;
    let offsets = tensor["data_offsets"].as_array().context(invalid)?;
    ensure!(
        offsets.len() == 2
            && offsets[0].as_u64() == Some(0)
            && offsets[1].as_u64() == Some(length as u64)
            && bytes.len() - start == length,
        "{invalid}"
    );
    let mut signal = false;
    for sample in bytes[start..].chunks_exact(width) {
        let (bits, exponent, magnitude) = if width == 4 {
            (
                u32::from_le_bytes(sample.try_into().unwrap()),
                0x7f80_0000,
                0x7fff_ffff,
            )
        } else {
            (
                u32::from(u16::from_le_bytes(sample.try_into().unwrap())),
                if dtype == "F16" { 0x7c00 } else { 0x7f80 },
                0x7fff,
            )
        };
        ensure!(bits & exponent != exponent, "{invalid}");
        signal |= bits & magnitude != 0;
    }
    ensure!(signal, "{invalid}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(dtype: &str, sample: &[u8]) -> Vec<u8> {
        let header = serde_json::json!({"speaker_wavs":{"dtype":dtype,"shape":[1,512,1],"data_offsets":[0,512*sample.len()]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        for _ in 0..512 {
            bytes.extend_from_slice(sample);
        }
        bytes
    }

    #[test]
    fn exported_voice_formats_reject_silence_nan_infinity_and_truncation() {
        for (dtype, good, bad) in [
            (
                "F32",
                1_f32.to_le_bytes().to_vec(),
                f32::NAN.to_le_bytes().to_vec(),
            ),
            (
                "F16",
                0x3c00_u16.to_le_bytes().to_vec(),
                0x7c00_u16.to_le_bytes().to_vec(),
            ),
            (
                "BF16",
                0x3f80_u16.to_le_bytes().to_vec(),
                0x7f80_u16.to_le_bytes().to_vec(),
            ),
        ] {
            let bytes = voice(dtype, &good);
            validate_voice(&bytes).unwrap();
            assert!(validate_voice(&bytes[..bytes.len() - 1]).is_err());
            assert!(validate_voice(&voice(dtype, &bad)).is_err());
            assert!(validate_voice(&voice(dtype, &vec![0; good.len()])).is_err());
        }
        assert!(validate_voice(&[255; 9]).is_err());
        assert!(validate_voice(&vec![0; MAX_VOICE_BYTES + 1]).is_err());
    }

    #[test]
    fn every_curated_asset_has_a_unique_safe_path_and_digest() {
        let mut paths = std::collections::HashSet::new();
        for asset in model_assets()
            .files
            .iter()
            .chain(voice_pack().voices.iter().map(|voice| &voice.file))
        {
            assert!(paths.insert(&asset.path));
            assert!(
                !asset.path.starts_with('/')
                    && !asset.path.contains("..")
                    && !asset.path.contains('\\')
            );
            assert_eq!(asset.sha256.len(), 64);
            assert!(asset.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert!(asset.size > 0);
        }
        assert_eq!(voice_pack().voices.len(), 37);
    }
}
