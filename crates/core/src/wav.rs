use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub fn wav_duration_seconds(path: &Path) -> Result<f64> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut byte_rate = None;
    let mut data_size = None;
    loop {
        let mut chunk = [0u8; 8];
        if file.read_exact(&mut chunk).is_err() {
            break;
        }
        let size = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
        match &chunk[0..4] {
            b"fmt " => {
                let mut fmt = vec![0u8; size as usize];
                file.read_exact(&mut fmt)?;
                if fmt.len() < 12 {
                    bail!("invalid fmt chunk");
                }
                byte_rate = Some(u32::from_le_bytes(fmt[8..12].try_into().unwrap()));
            }
            b"data" => {
                data_size = Some(size);
                file.seek(SeekFrom::Current(i64::from(size)))?;
            }
            _ => {
                file.seek(SeekFrom::Current(i64::from(size)))?;
            }
        }
        if size % 2 == 1 {
            file.seek(SeekFrom::Current(1))?;
        }
        if byte_rate.is_some() && data_size.is_some() {
            break;
        }
    }
    let rate = byte_rate.context("WAV has no fmt chunk")?;
    if rate == 0 {
        bail!("WAV byte rate is zero");
    }
    Ok(f64::from(data_size.context("WAV has no data chunk")?) / f64::from(rate))
}
