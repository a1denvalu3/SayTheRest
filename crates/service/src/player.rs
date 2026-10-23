use crate::stretch::TimeStretch;
use anyhow::{Context as _, Result, ensure};
use rminiaudio::{Context, Device, DeviceConfig, DeviceType, Format};
use sayit_protocol::PlaybackTextChunk;
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};
use uuid::Uuid;

#[derive(Clone)]
pub struct PlaybackController {
    tx: mpsc::Sender<Command>,
    shared: Arc<Mutex<AudioBuffer>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerStatus {
    Idle,
    Buffering,
    Playing,
    Paused,
    Finished,
    Failed,
}

#[derive(Clone, Debug)]
pub struct PlayerSnapshot {
    pub status: PlayerStatus,
    pub job_id: Option<Uuid>,
    pub position_seconds: f64,
    pub duration_seconds: f64,
    pub rate: f64,
    pub volume: f64,
    pub generation_complete: bool,
    pub amplitudes: Vec<f32>,
    pub error: Option<String>,
    pub spoken_chunks: Vec<PlaybackTextChunk>,
}

/// Retains original mono PCM for seeking. The device callback consumes ready
/// PCM without DSP or file I/O. Underruns buffer instead of completing playback.
struct AudioBuffer {
    source: Vec<f32>,
    rendered: VecDeque<f32>,
    cursor: f64,
    sample_rate: u32,
    status: PlayerStatus,
    job_id: Option<Uuid>,
    rate: f64,
    volume: f64,
    complete: bool,
    drained: bool,
    error: Option<String>,
    amplitudes: Vec<f32>,
    amplitude_energy: f64,
    amplitude_frames: usize,
    spoken_chunks: Vec<PlaybackTextChunk>,
}
impl Default for AudioBuffer {
    fn default() -> Self {
        Self {
            source: Vec::new(),
            rendered: VecDeque::new(),
            cursor: 0.0,
            sample_rate: 0,
            status: PlayerStatus::Idle,
            job_id: None,
            rate: 1.0,
            volume: 1.0,
            complete: false,
            drained: false,
            error: None,
            amplitudes: Vec::new(),
            amplitude_energy: 0.0,
            amplitude_frames: 0,
            spoken_chunks: Vec::new(),
        }
    }
}
impl AudioBuffer {
    fn begin(&mut self, id: Uuid, rate: f64, volume: f64) {
        *self = Self {
            job_id: Some(id),
            rate,
            volume,
            status: PlayerStatus::Buffering,
            ..Self::default()
        };
    }
    fn append(&mut self, id: Uuid, sample_rate: u32, samples: &[f32]) -> Result<()> {
        ensure!(self.job_id == Some(id), "speech was stopped or replaced");
        ensure!(!self.complete, "speech stream already completed");
        ensure!(self.error.is_none(), "audio playback failed");
        ensure!(
            (8_000..=192_000).contains(&sample_rate),
            "unsupported audio sample rate"
        );
        ensure!(
            self.sample_rate == 0 || self.sample_rate == sample_rate,
            "audio sample rate changed during speech"
        );
        ensure!(
            samples.iter().all(|x| x.is_finite()),
            "non-finite audio samples"
        );
        self.sample_rate = sample_rate;
        self.source.extend_from_slice(samples);
        for sample in samples {
            self.amplitude_energy += f64::from(*sample).powi(2);
            self.amplitude_frames += 1;
            if self.amplitude_frames == (sample_rate / 20) as usize {
                self.amplitudes
                    .push((self.amplitude_energy / self.amplitude_frames as f64).sqrt() as f32);
                self.amplitude_energy = 0.0;
                self.amplitude_frames = 0;
            }
        }
        Ok(())
    }
    fn render_into(&mut self, output: &mut [f32]) {
        output.fill(0.0);
        if self.status != PlayerStatus::Playing {
            return;
        }
        for sample in output {
            let Some(value) = self.rendered.pop_front() else {
                self.status = if self.complete && self.drained {
                    PlayerStatus::Finished
                } else {
                    PlayerStatus::Buffering
                };
                if self.status == PlayerStatus::Finished {
                    self.cursor = self.source.len() as f64;
                }
                break;
            };
            *sample = value * self.volume as f32;
            self.cursor = (self.cursor + self.rate).min(self.source.len() as f64);
        }
    }
}

type Reply = mpsc::Sender<Result<()>>;
enum Command {
    TextBlock {
        id: Uuid,
        start: usize,
        end: usize,
        finished: bool,
        reply: Reply,
    },
    RestoreChunks {
        id: Uuid,
        chunks: Vec<PlaybackTextChunk>,
        reply: Reply,
    },
    Begin {
        id: Uuid,
        rate: f64,
        volume: f64,
        reply: Reply,
    },
    Append {
        id: Uuid,
        sample_rate: u32,
        samples: Vec<f32>,
        reply: Reply,
    },
    Finish {
        id: Uuid,
        reply: Reply,
    },
    Pause(Reply),
    Resume(Reply),
    Stop(Reply),
    Abort(Uuid),
    Seek(f64, Reply),
    SetRate(f64, Reply),
    SetVolume(f64, Reply),
}
impl PlaybackController {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Mutex::new(AudioBuffer::default()));
        let thread_shared = shared.clone();
        thread::Builder::new()
            .name("sayit-audio".into())
            .spawn(move || player_thread(rx, thread_shared))
            .expect("audio thread");
        Self { tx, shared }
    }
    fn request(&self, make: impl FnOnce(Reply) -> Command) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(make(tx))?;
        rx.recv().context("audio thread stopped")?
    }
    pub fn begin_stream(&self, id: Uuid, rate: f64, volume: f64) -> Result<()> {
        valid_rate(rate)?;
        valid_volume(volume)?;
        self.request(|reply| Command::Begin {
            id,
            rate,
            volume,
            reply,
        })
    }
    pub fn append(&self, id: Uuid, sample_rate: u32, samples: &[f32]) -> Result<()> {
        self.request(|reply| Command::Append {
            id,
            sample_rate,
            samples: samples.to_vec(),
            reply,
        })
    }
    pub fn finish_stream(&self, id: Uuid) -> Result<()> {
        self.request(|reply| Command::Finish { id, reply })
    }
    pub fn text_block(&self, id: Uuid, start: usize, end: usize, finished: bool) -> Result<()> {
        self.request(|reply| Command::TextBlock {
            id,
            start,
            end,
            finished,
            reply,
        })
    }
    pub fn restore_chunks(&self, id: Uuid, chunks: Vec<PlaybackTextChunk>) -> Result<()> {
        self.request(|reply| Command::RestoreChunks { id, chunks, reply })
    }
    pub fn abort_stream(&self, id: Uuid) -> Result<()> {
        self.tx.send(Command::Abort(id))?;
        Ok(())
    }
    pub fn play(&self, id: Uuid, path: PathBuf, rate: f64, volume: f64) -> Result<()> {
        let mut reader = hound::WavReader::open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let spec = reader.spec();
        ensure!(
            spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int
                && spec.channels > 0,
            "only 16-bit PCM WAV playback is currently supported"
        );
        self.begin_stream(id, rate, volume)?;
        let result = (|| -> Result<()> {
            let mut block = Vec::with_capacity(4096);
            let mut sum = 0.0;
            let mut channel = 0;
            for sample in reader.samples::<i16>() {
                sum += f32::from(sample?) / 32768.0;
                channel += 1;
                if channel == spec.channels {
                    block.push(sum / f32::from(spec.channels));
                    sum = 0.0;
                    channel = 0;
                    if block.len() == 4096 {
                        self.append(id, spec.sample_rate, &block)?;
                        block.clear();
                    }
                }
            }
            ensure!(channel == 0, "incomplete audio frame");
            if !block.is_empty() {
                self.append(id, spec.sample_rate, &block)?;
            }
            self.finish_stream(id)
        })();
        if result.is_err() {
            let _ = self.abort_stream(id);
        }
        result
    }
    pub fn pause(&self) -> Result<()> {
        self.request(Command::Pause)
    }
    pub fn resume(&self) -> Result<()> {
        self.request(Command::Resume)
    }
    pub fn stop(&self) -> Result<()> {
        self.request(Command::Stop)
    }
    pub fn seek(&self, seconds: f64) -> Result<()> {
        ensure!(seconds.is_finite(), "seconds must be finite");
        self.request(|reply| Command::Seek(seconds, reply))
    }
    pub fn set_rate(&self, rate: f64) -> Result<()> {
        valid_rate(rate)?;
        self.request(|reply| Command::SetRate(rate, reply))
    }
    pub fn set_volume(&self, volume: f64) -> Result<()> {
        valid_volume(volume)?;
        self.request(|reply| Command::SetVolume(volume, reply))
    }
    pub fn snapshot(&self) -> PlayerSnapshot {
        let state = self.shared.lock().unwrap();
        let sample_rate = f64::from(state.sample_rate.max(1));
        let stride = state.amplitudes.len().div_ceil(1024).max(1);
        let amplitudes = state
            .amplitudes
            .chunks(stride)
            .map(|chunk| chunk.iter().copied().fold(0.0, f32::max))
            .collect();
        PlayerSnapshot {
            status: state.status,
            job_id: state.job_id,
            position_seconds: state.cursor / sample_rate,
            duration_seconds: state.source.len() as f64 / sample_rate,
            rate: state.rate,
            volume: state.volume,
            generation_complete: state.complete,
            amplitudes,
            error: state.error.clone(),
            spoken_chunks: state.spoken_chunks.clone(),
        }
    }
}
fn valid_rate(rate: f64) -> Result<()> {
    ensure!(
        rate.is_finite() && (0.5..=3.0).contains(&rate),
        "playback rate must be between 0.5 and 3.0"
    );
    Ok(())
}
fn valid_volume(volume: f64) -> Result<()> {
    ensure!(
        volume.is_finite() && (0.0..=1.0).contains(&volume),
        "volume must be between 0 and 1"
    );
    Ok(())
}

#[derive(Default)]
struct Renderer {
    stretch: Option<TimeStretch>,
    next_input: usize,
    flushed: bool,
}
impl Renderer {
    fn reset(&mut self, state: &mut AudioBuffer) {
        *self = Self {
            next_input: state.cursor.floor() as usize,
            ..Self::default()
        };
        state.rendered.clear();
        state.drained = false;
        if !matches!(
            state.status,
            PlayerStatus::Idle | PlayerStatus::Paused | PlayerStatus::Failed
        ) {
            state.status = PlayerStatus::Buffering;
        }
    }
    fn pump(&mut self, shared: &Arc<Mutex<AudioBuffer>>) -> Result<()> {
        loop {
            let (input, final_block, sample_rate, rate) = {
                let state = shared.lock().unwrap();
                if state.sample_rate == 0
                    || self.flushed
                    || matches!(state.status, PlayerStatus::Idle | PlayerStatus::Failed)
                {
                    return Ok(());
                }
                if state.rendered.len() >= (state.sample_rate / 4) as usize {
                    return Ok(());
                }
                let end = (self.next_input + 4096).min(state.source.len());
                if end == self.next_input && !state.complete {
                    return Ok(());
                }
                (
                    state.source[self.next_input..end].to_vec(),
                    state.complete && end == state.source.len(),
                    state.sample_rate,
                    state.rate,
                )
            };
            if self.stretch.is_none() {
                self.stretch = Some(TimeStretch::new(sample_rate, rate)?);
            }
            // Do DSP outside the device callback's lock.
            let rendered = self
                .stretch
                .as_mut()
                .unwrap()
                .process(&input, final_block)?;
            self.next_input += input.len();
            self.flushed = final_block;
            let mut state = shared.lock().unwrap();
            state.rendered.extend(rendered);
            state.drained = final_block;
            if state.status == PlayerStatus::Buffering
                && (state.rendered.len() >= (sample_rate * 12 / 100) as usize
                    || (final_block && !state.rendered.is_empty()))
            {
                state.status = PlayerStatus::Playing;
            }
            if final_block && state.rendered.is_empty() && state.status != PlayerStatus::Paused {
                state.status = PlayerStatus::Finished;
                state.cursor = state.source.len() as f64;
            }
        }
    }
}
fn player_thread(rx: mpsc::Receiver<Command>, shared: Arc<Mutex<AudioBuffer>>) {
    let mut device: Option<Device> = None;
    let mut renderer = Renderer::default();
    loop {
        match rx.recv_timeout(Duration::from_millis(5)) {
            Ok(command) => match command {
                Command::TextBlock {
                    id,
                    start,
                    end,
                    finished,
                    reply,
                } => {
                    let mut state = shared.lock().unwrap();
                    let result = if state.job_id != Some(id) {
                        Err(anyhow::anyhow!("speech was stopped or replaced"))
                    } else {
                        let time = state.source.len() as f64 / f64::from(state.sample_rate.max(1));
                        if finished {
                            if let Some(chunk) = state.spoken_chunks.last_mut() {
                                chunk.audio_end_seconds = Some(time);
                            }
                        } else {
                            state.spoken_chunks.push(PlaybackTextChunk {
                                text_start: start,
                                text_end: end,
                                audio_start_seconds: time,
                                audio_end_seconds: None,
                            });
                        }
                        Ok(())
                    };
                    let _ = reply.send(result);
                }
                Command::RestoreChunks { id, chunks, reply } => {
                    let mut state = shared.lock().unwrap();
                    let result = if state.job_id == Some(id) {
                        state.spoken_chunks = chunks;
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("speech was stopped or replaced"))
                    };
                    let _ = reply.send(result);
                }
                Command::Begin {
                    id,
                    rate,
                    volume,
                    reply,
                } => {
                    device.take();
                    shared.lock().unwrap().begin(id, rate, volume);
                    renderer = Renderer::default();
                    let _ = reply.send(Ok(()));
                }
                Command::Append {
                    id,
                    sample_rate,
                    samples,
                    reply,
                } => {
                    let result = shared.lock().unwrap().append(id, sample_rate, &samples);
                    let result = result.and_then(|()| {
                        if device.is_none() {
                            device = Some(start_device(sample_rate, &shared)?);
                        }
                        Ok(())
                    });
                    if let Err(error) = &result {
                        let mut state = shared.lock().unwrap();
                        if state.job_id == Some(id) {
                            state.error = Some(error.to_string());
                            state.status = PlayerStatus::Failed;
                        }
                    }
                    let _ = reply.send(result);
                }
                Command::Finish { id, reply } => {
                    let mut state = shared.lock().unwrap();
                    let result = if state.job_id != Some(id) {
                        Err(anyhow::anyhow!("speech was stopped or replaced"))
                    } else if state.source.is_empty() {
                        Err(anyhow::anyhow!("inference generated empty audio"))
                    } else {
                        state.complete = true;
                        Ok(())
                    };
                    let _ = reply.send(result);
                }
                Command::Stop(reply) => {
                    device.take();
                    *shared.lock().unwrap() = AudioBuffer::default();
                    renderer = Renderer::default();
                    let _ = reply.send(Ok(()));
                }
                Command::Abort(id) => {
                    if shared.lock().unwrap().job_id == Some(id) {
                        device.take();
                        *shared.lock().unwrap() = AudioBuffer::default();
                        renderer = Renderer::default();
                    }
                }
                Command::Pause(reply) => {
                    let mut state = shared.lock().unwrap();
                    if state.job_id.is_some() {
                        state.status = PlayerStatus::Paused;
                    }
                    let _ = reply.send(Ok(()));
                }
                Command::Resume(reply) => {
                    let mut state = shared.lock().unwrap();
                    if state.status == PlayerStatus::Finished {
                        state.cursor = 0.0;
                        renderer.reset(&mut state);
                    }
                    if state.status == PlayerStatus::Paused {
                        state.status = PlayerStatus::Playing;
                    }
                    let _ = reply.send(Ok(()));
                }
                Command::Seek(seconds, reply) => {
                    let mut state = shared.lock().unwrap();
                    if state.status == PlayerStatus::Finished {
                        state.status = PlayerStatus::Paused;
                    }
                    state.cursor = (seconds.max(0.0) * f64::from(state.sample_rate))
                        .min(state.source.len() as f64);
                    renderer.reset(&mut state);
                    let _ = reply.send(Ok(()));
                }
                Command::SetRate(rate, reply) => {
                    let mut state = shared.lock().unwrap();
                    if state.rate != rate {
                        state.rate = rate;
                        renderer.reset(&mut state);
                    }
                    let _ = reply.send(Ok(()));
                }
                Command::SetVolume(volume, reply) => {
                    shared.lock().unwrap().volume = volume;
                    let _ = reply.send(Ok(()));
                }
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if let Err(error) = renderer.pump(&shared) {
            let mut state = shared.lock().unwrap();
            state.error = Some(format!("speech tempo processing failed: {error:#}"));
            state.status = PlayerStatus::Failed;
        }
    }
}
fn start_device(sample_rate: u32, shared: &Arc<Mutex<AudioBuffer>>) -> Result<Device> {
    let callback_state = shared.clone();
    let config = DeviceConfig::new(DeviceType::Playback)
        .sample_rate(sample_rate)
        .channels(1)
        .format(Format::F32)
        .data_callback(move |output, _| callback_state.lock().unwrap().render_into(output));
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
    fn acknowledged_controls_are_visible_to_immediate_snapshots() {
        let player = PlaybackController::new();
        let id = Uuid::new_v4();
        player.begin_stream(id, 1.0, 1.0).unwrap();
        // Exercise the real command thread without opening an audio device.
        {
            let mut state = player.shared.lock().unwrap();
            state.sample_rate = 24_000;
            state.source = vec![0.1; 48_000];
            state.complete = true;
            state.status = PlayerStatus::Finished;
        }
        player.seek(0.75).unwrap();
        assert_eq!(player.snapshot().position_seconds, 0.75);
        assert_eq!(player.snapshot().status, PlayerStatus::Paused);
        player.set_rate(1.5).unwrap();
        player.set_volume(0.25).unwrap();
        assert_eq!(player.snapshot().rate, 1.5);
        assert_eq!(player.snapshot().volume, 0.25);
        assert_eq!(player.snapshot().status, PlayerStatus::Paused);
        player.resume().unwrap();
        assert_eq!(player.snapshot().status, PlayerStatus::Playing);
        player.pause().unwrap();
        assert_eq!(player.snapshot().status, PlayerStatus::Paused);
        player.stop().unwrap();
        assert_eq!(player.snapshot().job_id, None);
    }
    #[test]
    fn rejects_invalid_controls_without_audio_device() {
        let player = PlaybackController::new();
        for rate in [0.1, f64::NAN, f64::INFINITY] {
            assert!(player.set_rate(rate).is_err());
        }
        for volume in [-1.0, 2.0, f64::NAN] {
            assert!(player.set_volume(volume).is_err());
        }
        assert!(player.seek(f64::NAN).is_err());
    }
    #[test]
    fn progressive_underrun_buffers_then_resumes_without_advancing_time() {
        let shared = Arc::new(Mutex::new(AudioBuffer::default()));
        let id = Uuid::new_v4();
        let mut renderer = Renderer::default();
        shared.lock().unwrap().begin(id, 1.0, 1.0);
        shared
            .lock()
            .unwrap()
            .append(id, 24_000, &vec![0.25; 4800])
            .unwrap();
        renderer.pump(&shared).unwrap();
        shared.lock().unwrap().render_into(&mut vec![0.0; 6000]);
        assert_eq!(shared.lock().unwrap().status, PlayerStatus::Buffering);
        assert_eq!(shared.lock().unwrap().cursor, 4800.0);
        shared.lock().unwrap().render_into(&mut [0.0; 256]);
        assert_eq!(shared.lock().unwrap().cursor, 4800.0);
        shared
            .lock()
            .unwrap()
            .append(id, 24_000, &vec![0.5; 4800])
            .unwrap();
        renderer.pump(&shared).unwrap();
        let mut output = [0.0; 256];
        shared.lock().unwrap().render_into(&mut output);
        assert!(output.iter().all(|x| *x == 0.5));
        shared.lock().unwrap().complete = true;
        renderer.pump(&shared).unwrap();
        shared.lock().unwrap().render_into(&mut vec![0.0; 6000]);
        assert_eq!(shared.lock().unwrap().status, PlayerStatus::Finished);
        assert_eq!(shared.lock().unwrap().cursor, 9600.0);
    }
    #[test]
    fn paused_stream_finishes_generation_without_resuming() {
        let shared = Arc::new(Mutex::new(AudioBuffer::default()));
        let id = Uuid::new_v4();
        shared.lock().unwrap().begin(id, 1.25, 0.5);
        shared.lock().unwrap().status = PlayerStatus::Paused;
        shared
            .lock()
            .unwrap()
            .append(id, 24_000, &vec![0.25; 2400])
            .unwrap();
        shared.lock().unwrap().complete = true;
        Renderer::default().pump(&shared).unwrap();
        let mut output = [1.0; 256];
        shared.lock().unwrap().render_into(&mut output);
        assert!(output.iter().all(|x| *x == 0.0));
        assert_eq!(shared.lock().unwrap().cursor, 0.0);
        assert_eq!(shared.lock().unwrap().status, PlayerStatus::Paused);
        shared.lock().unwrap().status = PlayerStatus::Playing;
        shared.lock().unwrap().render_into(&mut vec![0.0; 2400]);
        assert_eq!(shared.lock().unwrap().status, PlayerStatus::Finished);
    }
    #[test]
    fn seek_and_rate_change_discard_rendered_audio_and_preserve_source() {
        let shared = Arc::new(Mutex::new(AudioBuffer::default()));
        let id = Uuid::new_v4();
        shared.lock().unwrap().begin(id, 1.0, 1.0);
        let source = (0..24000)
            .map(|i| if i < 12000 { 0.1 } else { -0.1 })
            .collect::<Vec<_>>();
        shared.lock().unwrap().append(id, 24_000, &source).unwrap();
        shared.lock().unwrap().complete = true;
        let mut renderer = Renderer::default();
        renderer.pump(&shared).unwrap();
        {
            let mut state = shared.lock().unwrap();
            state.cursor = 15000.0;
            state.rate = 1.5;
            renderer.reset(&mut state);
            assert!(state.rendered.is_empty());
        }
        renderer.pump(&shared).unwrap();
        let mut output = [0.0; 1000];
        shared.lock().unwrap().render_into(&mut output);
        assert!(output.iter().all(|x| *x <= 0.0));
        assert_eq!(shared.lock().unwrap().cursor, 16500.0);
        assert_eq!(shared.lock().unwrap().source, source);
    }
    #[test]
    fn stopped_replaced_and_invalid_audio_cannot_reenter_player() {
        let mut state = AudioBuffer::default();
        let old = Uuid::new_v4();
        let current = Uuid::new_v4();
        state.begin(current, 1.0, 1.0);
        assert!(state.append(old, 24_000, &[0.1]).is_err());
        assert!(state.append(current, 24_000, &[f32::NAN]).is_err());
        assert!(state.append(current, 0, &[0.1]).is_err());
        state.append(current, 24_000, &[0.1]).unwrap();
        assert!(state.append(current, 48_000, &[0.1]).is_err());
        state.complete = true;
        assert!(state.append(current, 24_000, &[0.1]).is_err());
        assert_eq!(state.source, [0.1]);
    }
}
