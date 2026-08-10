use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceEvent {
    pub sequence: u64,
    pub resource: String,
    pub occurred_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Health {
    pub status: &'static str,
    pub protocol_version: u32,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueuePolicy {
    #[default]
    Replace,
    Append,
    Interrupt,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SpeechSubmission {
    pub text: String,
    #[serde(default)]
    pub voice_profile_id: Option<Uuid>,
    #[serde(default)]
    pub source: SpeechSource,
    #[serde(default)]
    pub queue_policy: QueuePolicy,
    #[serde(default)]
    pub confirmed_long_text: bool,
    #[serde(default)]
    pub speaking_pace: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechSource {
    Selection,
    Clipboard,
    Cli,
    Api,
    History,
    #[default]
    Desktop,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Synthesizing,
    Playing,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpeechJob {
    pub id: Uuid,
    pub text: String,
    pub source: SpeechSource,
    pub state: JobState,
    pub created_at: DateTime<Utc>,
    pub error: Option<String>,
    #[serde(default)]
    pub voice_profile_id: Option<Uuid>,
    #[serde(default)]
    pub speaking_pace: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlaybackSnapshot {
    pub state: Option<JobState>,
    pub job_id: Option<Uuid>,
    pub position_seconds: f64,
    pub duration_seconds: f64,
    pub rate: f64,
    pub volume: f64,
    #[serde(default)]
    pub current_title: String,
    #[serde(default)]
    pub spoken_text: String,
    #[serde(default)]
    pub spoken_chunks: Vec<PlaybackTextChunk>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlaybackTextChunk {
    /// Unicode character offset in `spoken_text`.
    pub text_start: usize,
    /// Exclusive Unicode character offset in `spoken_text`.
    pub text_end: usize,
    pub audio_start_seconds: f64,
}

impl Default for PlaybackSnapshot {
    fn default() -> Self {
        Self {
            state: None,
            job_id: None,
            position_seconds: 0.0,
            duration_seconds: 0.0,
            rate: 1.0,
            volume: 1.0,
            current_title: String::new(),
            spoken_text: String::new(),
            spoken_chunks: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HistoryItem {
    pub job: SpeechJob,
    pub audio_path: Option<String>,
    pub duration_seconds: Option<f64>,
    pub pinned: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceSettings {
    pub active_model_id: Option<String>,
    #[serde(default)]
    pub active_voice_profile_id: Option<Uuid>,
    #[serde(default)]
    pub voice_profile_by_model: HashMap<String, Option<Uuid>>,
    pub playback_rate: f64,
    #[serde(default = "natural_speaking_pace")]
    pub speaking_pace: f64,
    pub volume: f64,
    pub model_unload_delay_seconds: u64,
    pub history_quota_bytes: u64,
    /// Maximum age of unpinned history in days. Zero keeps history indefinitely.
    #[serde(default)]
    pub history_retention_days: u32,
    pub long_text_confirmation_characters: usize,
    #[serde(default)]
    pub text_cleaning: TextCleaningSettings,
}

impl Default for ServiceSettings {
    fn default() -> Self {
        Self {
            active_model_id: Some("piper-en-us-lessac-medium".into()),
            active_voice_profile_id: None,
            voice_profile_by_model: HashMap::new(),
            playback_rate: 1.0,
            speaking_pace: 1.0,
            volume: 1.0,
            model_unload_delay_seconds: 600,
            history_quota_bytes: 1_073_741_824,
            history_retention_days: 0,
            long_text_confirmation_characters: 20_000,
            text_cleaning: TextCleaningSettings::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ServiceSettingsUpdate {
    pub playback_rate: Option<f64>,
    pub speaking_pace: Option<f64>,
    pub volume: Option<f64>,
    pub model_unload_delay_seconds: Option<u64>,
    pub history_quota_bytes: Option<u64>,
    pub history_retention_days: Option<u32>,
    pub long_text_confirmation_characters: Option<usize>,
    pub text_cleaning: Option<TextCleaningSettings>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct TextCleaningSettings {
    pub enabled: bool,
    pub strip_markdown: bool,
    pub strip_html: bool,
    pub strip_code_blocks: bool,
    pub strip_special_characters: bool,
    pub normalize_whitespace: bool,
}

impl Default for TextCleaningSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            strip_markdown: true,
            strip_html: true,
            strip_code_blocks: true,
            strip_special_characters: true,
            normalize_whitespace: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub name: String,
    pub languages: Vec<String>,
    pub size_bytes: u64,
    pub license: String,
    pub license_url: String,
    pub installed: bool,
    #[serde(default)]
    pub update_available: bool,
    #[serde(default)]
    pub resident: bool,
    pub selected: bool,
    pub capabilities: ModelCapabilities,
    pub download: Option<DownloadSnapshot>,
    pub quality_note: String,
    pub speed_note: String,
    pub benchmark: Option<ModelBenchmark>,
    pub recommended: bool,
    #[serde(default)]
    pub supports_native_speaking_pace: bool,
    #[serde(default)]
    pub preset_voices: Vec<ModelPresetVoice>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelPresetVoice {
    pub id: String,
    pub name: String,
    pub language: String,
    pub selected: bool,
}

fn natural_speaking_pace() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelBenchmark {
    pub iterations: usize,
    pub cold_start_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_rtf: f64,
    pub audio_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LocalModelImportRequest {
    pub directory: String,
    pub display_name: String,
    pub license: String,
    pub license_url: String,
    pub untested_model_and_license_review_confirmed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HuggingFaceModelImportRequest {
    pub repository: String,
    pub revision: Option<String>,
    pub display_name: String,
    pub license: String,
    pub license_url: String,
    pub access_token: Option<String>,
    pub untested_model_and_license_review_confirmed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCapabilities {
    pub preset_voices: bool,
    pub voice_cloning: bool,
    pub voice_description: bool,
    pub streaming: bool,
    pub long_form: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Downloading,
    Verifying,
    Installing,
    Installed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DownloadSnapshot {
    pub state: DownloadState,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VoiceCloneRequest {
    pub name: String,
    pub reference_audio_path: String,
    pub speaker_permission_confirmed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VoiceProfile {
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub model_id: String,
    pub reference_audio_path: String,
    pub reference_duration_seconds: f64,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub quality: VoiceQuality,
    #[serde(default = "default_voice_refinement_steps")]
    pub refinement_steps: u32,
}

fn default_voice_refinement_steps() -> u32 {
    5
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct VoiceQuality {
    pub sample_rate: u32,
    pub original_duration_seconds: f64,
    pub trimmed_duration_seconds: f64,
    pub peak_dbfs: f64,
    pub rms_dbfs: f64,
    pub clipped_sample_percent: f64,
    pub quality_score: u8,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VoiceRenameRequest {
    pub name: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct VoiceTuningRequest {
    pub refinement_steps: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct SecondsRequest {
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct PlaybackRateRequest {
    pub rate: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct VolumeRequest {
    pub volume: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct PinRequest {
    pub pinned: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceSnapshot {
    pub protocol_version: u32,
    pub active_model_id: Option<String>,
    pub queue_depth: usize,
    pub playback: PlaybackSnapshot,
}
