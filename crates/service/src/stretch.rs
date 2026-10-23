//! Continuous pitch-preserving speech tempo processing. Only the audio worker
//! owns this handle; the device callback only consumes already rendered PCM.
use anyhow::{Result, ensure};
use std::{collections::VecDeque, ffi::c_void, ptr::NonNull};

unsafe extern "C" {
    fn sonicCreateStream(sample_rate: i32, channels: i32) -> *mut c_void;
    fn sonicDestroyStream(stream: *mut c_void);
    fn sonicSetSpeed(stream: *mut c_void, speed: f32);
    fn sonicSetQuality(stream: *mut c_void, quality: i32);
    fn sonicWriteFloatToStream(stream: *mut c_void, samples: *const f32, frames: i32) -> i32;
    fn sonicReadFloatFromStream(stream: *mut c_void, samples: *mut f32, frames: i32) -> i32;
    fn sonicSamplesAvailable(stream: *mut c_void) -> i32;
    fn sonicFlushStream(stream: *mut c_void) -> i32;
}

pub(crate) struct TimeStretch {
    handle: NonNull<c_void>,
    rate: f64,
    sample_rate: u32,
    input_frames: usize,
    delivered_frames: usize,
    pending: VecDeque<f32>,
    finished: bool,
}

impl TimeStretch {
    pub fn new(sample_rate: u32, rate: f64) -> Result<Self> {
        ensure!(
            (8_000..=192_000).contains(&sample_rate),
            "unsupported sample rate"
        );
        ensure!(
            rate.is_finite() && (0.5..=3.0).contains(&rate),
            "invalid playback rate"
        );
        // SAFETY: Valid format, checked allocation; this object uniquely owns
        // the stream until Drop. Sonic defaults pitch and resampling rate to 1.
        let handle = NonNull::new(unsafe { sonicCreateStream(sample_rate as i32, 1) })
            .ok_or_else(|| anyhow::anyhow!("could not allocate speech time stretcher"))?;
        unsafe {
            sonicSetSpeed(handle.as_ptr(), rate as f32);
            sonicSetQuality(handle.as_ptr(), 1);
        }
        Ok(Self {
            handle,
            rate,
            sample_rate,
            input_frames: 0,
            delivered_frames: 0,
            pending: VecDeque::new(),
            finished: false,
        })
    }

    pub fn process(&mut self, samples: &[f32], final_block: bool) -> Result<Vec<f32>> {
        ensure!(!self.finished, "time stretcher already finalized");
        ensure!(
            samples.iter().all(|sample| sample.is_finite()),
            "non-finite audio sample"
        );
        self.input_frames += samples.len();
        self.finished = final_block;
        if self.rate == 1.0 {
            self.delivered_frames += samples.len();
            return Ok(samples.to_vec());
        }
        // Sonic's float interface converts internally to i16. Bound hot PCM
        // before conversion. The source WAV and exported audio stay untouched.
        let bounded = samples
            .iter()
            .map(|x| x.clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        self.write(&bounded)?;
        if final_block {
            self.write(&vec![0.0; (self.sample_rate / 5) as usize])?;
            ensure!(
                unsafe { sonicFlushStream(self.handle.as_ptr()) } != 0,
                "could not flush speech tempo processor"
            );
        }
        let available = unsafe { sonicSamplesAvailable(self.handle.as_ptr()) };
        ensure!(available >= 0, "invalid speech tempo output");
        let mut output = vec![0.0; available as usize];
        if available > 0 {
            let read = unsafe {
                sonicReadFloatFromStream(self.handle.as_ptr(), output.as_mut_ptr(), available)
            };
            ensure!(read == available, "incomplete speech tempo output");
            self.pending.extend(output);
        }
        let expected = (self.input_frames as f64 / self.rate).round() as usize;
        let count = self
            .pending
            .len()
            .min(expected.saturating_sub(self.delivered_frames));
        let output = self.pending.drain(..count).collect();
        self.delivered_frames += count;
        if final_block {
            ensure!(
                self.delivered_frames == expected,
                "speech tempo processor lost the audio tail"
            );
        }
        Ok(output)
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        let frames = i32::try_from(samples.len())?;
        ensure!(
            unsafe { sonicWriteFloatToStream(self.handle.as_ptr(), samples.as_ptr(), frames) } != 0,
            "could not process speech tempo"
        );
        Ok(())
    }
}

impl Drop for TimeStretch {
    fn drop(&mut self) {
        unsafe { sonicDestroyStream(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn tone(sample_rate: u32, frequency: f32, seconds: f32) -> Vec<f32> {
        (0..(sample_rate as f32 * seconds) as usize)
            .map(|i| 0.6 * (TAU * frequency * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn streaming_tempo_preserves_pitch_duration_and_tail() {
        for sample_rate in [16_000, 24_000, 44_100, 48_000] {
            for rate in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0] {
                let source = tone(sample_rate, 220.0, 2.0);
                let mut stretch = TimeStretch::new(sample_rate, rate).unwrap();
                let mut output = Vec::new();
                for chunk in source.chunks(997) {
                    output.extend(stretch.process(chunk, false).unwrap());
                }
                assert!(!output.is_empty(), "must emit before completion at {rate}");
                output.extend(stretch.process(&[], true).unwrap());
                assert_eq!(output.len(), (source.len() as f64 / rate).round() as usize);
                assert!(output.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
                let middle = &output[output.len() / 4..output.len() * 3 / 4];
                let crossings = middle
                    .windows(2)
                    .filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
                    .count();
                let frequency = crossings as f64 * sample_rate as f64 / middle.len() as f64;
                assert!(
                    (frequency - 220.0).abs() < 5.0,
                    "pitch changed at {rate} and {sample_rate} Hz sample rate: {frequency}"
                );
                if rate == 1.0 {
                    assert_eq!(source, output);
                }
            }
        }
    }

    #[test]
    fn short_hot_and_silent_input_are_bounded_and_fully_drained() {
        for source in [vec![0.0; 7], vec![1.5; 300], tone(24_000, 330.0, 0.017)] {
            for rate in [0.5, 1.25, 3.0] {
                let mut stretch = TimeStretch::new(24_000, rate).unwrap();
                let output = stretch.process(&source, true).unwrap();
                assert_eq!(output.len(), (source.len() as f64 / rate).round() as usize);
                assert!(output.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
                assert!(stretch.process(&[], true).is_err());
            }
        }
        assert!(TimeStretch::new(0, 1.0).is_err());
        assert!(TimeStretch::new(24_000, f64::NAN).is_err());
    }
}
