use rminiaudio::{Context, Device, DeviceConfig, DeviceType, Format};
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SAMPLE_RATE: u32 = 16_000;
const MAX_SECONDS: usize = 15;

pub struct RecorderState(pub Mutex<Option<ActiveRecording>>);

pub struct ActiveRecording {
    device: Device,
    samples: Arc<Mutex<Vec<f32>>>,
    started: Instant,
}

#[derive(Serialize)]
pub struct RecordingStatus {
    pub recording: bool,
    pub elapsed_seconds: f64,
    pub maximum_seconds: usize,
}

#[derive(Serialize)]
pub struct RecordingResult {
    pub path: String,
    pub duration_seconds: f64,
    pub sample_rate: u32,
}

pub fn start(state: &RecorderState) -> Result<RecordingStatus, String> {
    let mut active = state.0.lock().map_err(|_| "recorder lock is unavailable")?;
    if active.is_some() {
        return Err("a voice recording is already in progress".into());
    }
    let samples = Arc::new(Mutex::new(Vec::with_capacity(
        SAMPLE_RATE as usize * MAX_SECONDS,
    )));
    let captured = Arc::clone(&samples);
    let config = DeviceConfig::new(DeviceType::Capture)
        .sample_rate(SAMPLE_RATE)
        .channels(1)
        .format(Format::F32)
        .data_callback(move |_output, input| {
            if let Ok(mut samples) = captured.try_lock() {
                let remaining = SAMPLE_RATE as usize * MAX_SECONDS - samples.len();
                samples.extend(input.iter().take(remaining).copied());
            }
        });
    let context =
        Context::new().map_err(|error| format!("cannot initialize audio capture: {error}"))?;
    let device = context
        .create_device(config)
        .map_err(|error| format!("cannot open the default microphone: {error}"))?;
    device
        .start()
        .map_err(|error| format!("cannot start microphone capture: {error}"))?;
    *active = Some(ActiveRecording {
        device,
        samples,
        started: Instant::now(),
    });
    Ok(status_from(active.as_ref()))
}

pub fn status(state: &RecorderState) -> Result<RecordingStatus, String> {
    let active = state.0.lock().map_err(|_| "recorder lock is unavailable")?;
    Ok(status_from(active.as_ref()))
}

pub fn cancel(state: &RecorderState) -> Result<(), String> {
    let mut active = state.0.lock().map_err(|_| "recorder lock is unavailable")?;
    if let Some(recording) = active.take() {
        let _ = recording.device.stop();
    }
    Ok(())
}

pub fn stop(state: &RecorderState, directory: &Path) -> Result<RecordingResult, String> {
    let recording = state
        .0
        .lock()
        .map_err(|_| "recorder lock is unavailable")?
        .take()
        .ok_or("no voice recording is in progress")?;
    recording
        .device
        .stop()
        .map_err(|error| format!("cannot stop microphone capture: {error}"))?;
    let samples = recording
        .samples
        .lock()
        .map_err(|_| "captured audio is unavailable")?;
    if samples.is_empty() {
        return Err("the microphone did not provide any audio".into());
    }
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create recording directory: {error}"))?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = directory.join(format!("voice-{timestamp}.wav"));
    write_pcm_wav(&path, &samples)?;
    Ok(RecordingResult {
        path: path.display().to_string(),
        duration_seconds: samples.len() as f64 / SAMPLE_RATE as f64,
        sample_rate: SAMPLE_RATE,
    })
}

fn status_from(active: Option<&ActiveRecording>) -> RecordingStatus {
    RecordingStatus {
        recording: active.is_some(),
        elapsed_seconds: active
            .map(|recording| {
                recording
                    .started
                    .elapsed()
                    .min(Duration::from_secs(MAX_SECONDS as u64))
                    .as_secs_f64()
            })
            .unwrap_or_default(),
        maximum_seconds: MAX_SECONDS,
    }
}

fn write_pcm_wav(path: &PathBuf, samples: &[f32]) -> Result<(), String> {
    let specification = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, specification)
        .map_err(|error| format!("cannot create recording: {error}"))?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer
            .write_sample(value)
            .map_err(|error| format!("cannot write recording: {error}"))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("cannot finalize recording: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_wav_is_voice_import_compatible_pcm() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sample.wav");
        write_pcm_wav(&path, &vec![0.25; SAMPLE_RATE as usize * 3]).unwrap();
        let reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.duration(), SAMPLE_RATE * 3);
    }
}
