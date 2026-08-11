use anyhow::{Context as _, Result, bail};
use rminiaudio::{Context, Device, DeviceConfig, DeviceType, Format};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
};
use uuid::Uuid;

#[derive(Clone)]
pub struct PlaybackController {
    tx: mpsc::Sender<Command>,
    shared: Arc<Mutex<Shared>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerStatus {
    Idle,
    Playing,
    Paused,
    Finished,
}

#[derive(Clone, Debug)]
pub struct PlayerSnapshot {
    pub status: PlayerStatus,
    pub job_id: Option<Uuid>,
    pub position_seconds: f64,
    pub duration_seconds: f64,
    pub rate: f64,
    pub volume: f64,
}

struct Shared {
    samples: Vec<f32>,
    cursor: f64,
    sample_rate: u32,
    channels: usize,
    status: PlayerStatus,
    job_id: Option<Uuid>,
    rate: f64,
    volume: f64,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            cursor: 0.0,
            sample_rate: 1,
            channels: 1,
            status: PlayerStatus::Idle,
            job_id: None,
            rate: 1.0,
            volume: 1.0,
        }
    }
}

enum Command {
    Play {
        job_id: Uuid,
        path: PathBuf,
        rate: f64,
        volume: f64,
        reply: mpsc::Sender<Result<()>>,
    },
    Pause,
    Resume,
    Stop,
    Seek(f64),
    SetRate(f64),
    SetVolume(f64),
}

impl PlaybackController {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let thread_shared = shared.clone();
        thread::Builder::new()
            .name("sayit-audio".into())
            .spawn(move || player_thread(rx, thread_shared))
            .expect("audio thread");
        Self { tx, shared }
    }

    pub fn play(&self, job_id: Uuid, path: PathBuf, rate: f64, volume: f64) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Command::Play {
            job_id,
            path,
            rate,
            volume,
            reply: tx,
        })?;
        rx.recv().context("audio thread stopped")?
    }
    pub fn pause(&self) -> Result<()> {
        self.tx.send(Command::Pause)?;
        Ok(())
    }
    pub fn resume(&self) -> Result<()> {
        self.tx.send(Command::Resume)?;
        Ok(())
    }
    pub fn stop(&self) -> Result<()> {
        self.tx.send(Command::Stop)?;
        Ok(())
    }
    pub fn seek(&self, seconds: f64) -> Result<()> {
        self.tx.send(Command::Seek(seconds))?;
        Ok(())
    }
    pub fn set_rate(&self, rate: f64) -> Result<()> {
        if !(0.5..=3.0).contains(&rate) {
            bail!("playback rate must be between 0.5 and 3.0");
        }
        self.tx.send(Command::SetRate(rate))?;
        Ok(())
    }
    pub fn set_volume(&self, volume: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&volume) {
            bail!("volume must be between 0 and 1");
        }
        self.tx.send(Command::SetVolume(volume))?;
        Ok(())
    }
    pub fn snapshot(&self) -> PlayerSnapshot {
        let state = self.shared.lock().unwrap();
        let total_frames = state.samples.len() / state.channels;
        PlayerSnapshot {
            status: state.status,
            job_id: state.job_id,
            position_seconds: state.cursor / f64::from(state.sample_rate),
            duration_seconds: total_frames as f64 / f64::from(state.sample_rate),
            rate: state.rate,
            volume: state.volume,
        }
    }
}

fn player_thread(rx: mpsc::Receiver<Command>, shared: Arc<Mutex<Shared>>) {
    let mut device: Option<Device> = None;
    while let Ok(command) = rx.recv() {
        match command {
            Command::Play {
                job_id,
                path,
                rate,
                volume,
                reply,
            } => {
                device.take();
                let result = load_and_start(job_id, path, rate, volume, &shared);
                match result {
                    Ok(new_device) => {
                        device = Some(new_device);
                        let _ = reply.send(Ok(()));
                    }
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Pause => {
                if let Some(device) = &device {
                    let _ = device.stop();
                }
                shared.lock().unwrap().status = PlayerStatus::Paused;
            }
            Command::Resume => {
                if let Some(device) = &device {
                    let _ = device.start();
                }
                shared.lock().unwrap().status = PlayerStatus::Playing;
            }
            Command::Stop => {
                device.take();
                let mut state = shared.lock().unwrap();
                state.status = PlayerStatus::Idle;
                state.cursor = 0.0;
                state.job_id = None;
            }
            Command::Seek(seconds) => {
                let mut state = shared.lock().unwrap();
                let frames = (seconds.max(0.0) * f64::from(state.sample_rate))
                    .min((state.samples.len() / state.channels) as f64);
                state.cursor = frames;
            }
            Command::SetRate(rate) => shared.lock().unwrap().rate = rate,
            Command::SetVolume(volume) => shared.lock().unwrap().volume = volume,
        }
    }
}

fn load_and_start(
    job_id: Uuid,
    path: PathBuf,
    rate: f64,
    volume: f64,
    shared: &Arc<Mutex<Shared>>,
) -> Result<Device> {
    let mut reader = hound::WavReader::open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
        bail!("only 16-bit PCM WAV playback is currently supported");
    }
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|value| f32::from(value) / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    {
        let mut state = shared.lock().unwrap();
        *state = Shared {
            samples,
            cursor: 0.0,
            sample_rate: spec.sample_rate,
            channels: usize::from(spec.channels),
            status: PlayerStatus::Playing,
            job_id: Some(job_id),
            rate,
            volume,
        };
    }
    let callback_state = shared.clone();
    let channels = usize::from(spec.channels);
    let config = DeviceConfig::new(DeviceType::Playback)
        .sample_rate(spec.sample_rate)
        .channels(u32::from(spec.channels))
        .format(Format::F32)
        .data_callback(move |output, _| {
            let mut state = callback_state.lock().unwrap();
            output.fill(0.0);
            if state.status != PlayerStatus::Playing {
                return;
            }
            for frame in output.chunks_exact_mut(channels) {
                let source_frame = state.cursor.floor() as usize;
                let offset = source_frame * channels;
                if offset + channels > state.samples.len() {
                    state.status = PlayerStatus::Finished;
                    break;
                }
                for channel in 0..channels {
                    frame[channel] = state.samples[offset + channel] * state.volume as f32;
                }
                state.cursor += state.rate;
            }
        });
    let context = Context::new().context("failed to initialize an audio backend")?;
    let device = context
        .create_device(config)
        .context("failed to open the default audio device")?;
    device.start().context("failed to start audio playback")?;
    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_rate_and_volume_without_audio_device() {
        let player = PlaybackController::new();
        assert!(player.set_rate(0.1).is_err());
        assert!(player.set_volume(2.0).is_err());
    }
}
