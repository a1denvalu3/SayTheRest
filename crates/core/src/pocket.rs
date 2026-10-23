//! Public, pinned source recordings for Pocket's upstream preset voices.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const MODEL_ARCHIVE_SHA256: &str =
    "2f3b88823cbbb9bf0b2477ec8ae7b3fec417b3a87b6bb5f256dba66f2ad967cb";

#[derive(Debug, Deserialize)]
pub struct VoiceAsset {
    pub id: String,
    pub name: String,
    pub language: String,
    pub path: String,
    pub source_path: String,
    pub size: u64,
    pub sha256: String,
    pub license: String,
    pub license_url: String,
    pub attribution: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
pub struct VoicePack {
    pub repository: String,
    pub revision: String,
    pub voices: Vec<VoiceAsset>,
    pub files: Vec<ModelFile>,
}

pub fn voice_pack() -> &'static VoicePack {
    static PACK: OnceLock<VoicePack> = OnceLock::new();
    PACK.get_or_init(|| {
        serde_json::from_str(include_str!("../data/pocket-voices.json"))
            .expect("bundled Pocket voice catalog")
    })
}

pub fn manifest_fingerprint() -> String {
    let mut hash = Sha256::new();
    hash.update(MODEL_ARCHIVE_SHA256.as_bytes());
    hash.update(include_bytes!("../data/pocket-voices.json"));
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_upstream_preset_has_a_pinned_recording_and_attribution() {
        let raw: serde_json::Value =
            serde_json::from_str(include_str!("../data/pocket-voices.json")).unwrap();
        assert_eq!(raw["archive"]["sha256"], MODEL_ARCHIVE_SHA256);
        let pack = voice_pack();
        assert_eq!(pack.voices.len(), 8);
        assert_eq!(pack.files.len(), 9);
        let mut paths = std::collections::HashSet::new();
        for file in &pack.files {
            assert!(paths.insert(&file.path));
            assert!(!file.path.contains(['/', '\\']) && !file.path.contains(".."));
            assert!(file.size > 0);
            assert_eq!(file.sha256.len(), 64);
            assert!(file.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        let mut ids = std::collections::HashSet::new();
        for voice in &pack.voices {
            assert!(ids.insert(&voice.id));
            assert_eq!(voice.path, format!("voices/{}.wav", voice.id));
            assert!(!voice.source_path.contains("..") && !voice.source_path.starts_with('/'));
            assert_eq!(voice.sha256.len(), 64);
            assert!(voice.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(voice.size > 44 && voice.size < 8 * 1024 * 1024);
            assert!(!voice.license.is_empty() && !voice.attribution.is_empty());
        }
    }
}
