#!/usr/bin/env python3
"""Review synthetic WAVs with a locally installed Whisper model, without network access.

The input is a JSON array of {audio, text, language, ...} samples. Audio paths
are relative to the manifest. Each sample must be mono PCM16 and at most 30 s;
longer recordings must be split explicitly so the recognizer cannot silently
truncate them. Requires numpy, torch, and transformers in the chosen interpreter.
This produces diagnostic evidence, not a perceptual quality certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata
import wave


def normalize_text(text):
    # Vowel signs and other combining marks carry pronunciation information.
    # isalnum() alone silently discards them in Hindi and several other scripts.
    return ''.join(char for char in unicodedata.normalize('NFC', text.casefold())
                   if char.isalnum() or unicodedata.category(char).startswith('M'))


def character_error_rate(expected, recognized):
    left, right = normalize_text(expected), normalize_text(recognized)
    if not left:
        raise ValueError('expected text must contain letters, numbers, or marks')
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        row = [index]
        for column, other in enumerate(right, 1):
            row.append(min(row[-1] + 1, previous[column] + 1,
                           previous[column - 1] + (char != other)))
        previous = row
    return previous[-1] / len(left)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_samples(manifest):
    samples = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(samples, list) or not samples:
        raise ValueError('sample manifest must be a nonempty JSON array')
    for sample in samples:
        if not isinstance(sample, dict) or not all(isinstance(sample.get(key), str)
                                                  for key in ['audio', 'text', 'language']):
            raise ValueError('each sample needs audio, text, and language strings')
        if not sample['language'].strip() or len(sample['text']) > 8192:
            raise ValueError('language is required and expected text must be at most 8192 characters')
        character_error_rate(sample['text'], '')
        sample = dict(sample)
        sample['audio'] = str((manifest.parent / sample['audio']).resolve())
        yield sample


def read_audio(path):
    # Validate the header before reading. ASR input is always explicitly bounded.
    with wave.open(str(path)) as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        if (audio.getnchannels() != 1 or audio.getsampwidth() != 2
                or audio.getcomptype() != 'NONE' or not 8000 <= rate <= 96000
                or not .1 <= frames / rate <= 30):
            raise ValueError(f'{path}: use mono PCM16 audio between 0.1 and 30 seconds')
        pcm = audio.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError(f'{path}: truncated WAV data')
    return rate, frames, pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True, help='JSON sample manifest')
    parser.add_argument('--model', type=Path, required=True, help='Local Whisper safetensors snapshot directory')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error('--threads must be between 1 and 64')
    model_path = args.model.resolve(strict=True)
    samples = list(read_samples(args.samples.resolve(strict=True)))
    # Preflight every input before loading the larger model or reporting results.
    for sample in samples:
        read_audio(Path(sample['audio']))
    weights = model_path / 'model.safetensors'
    if not weights.is_file():
        parser.error('--model must contain model.safetensors')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(args.threads)
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True).eval()
    assets = {path.name: sha256(path) for path in sorted(model_path.iterdir())
              if path.is_file() and (path.suffix in {'.json', '.txt'} or path == weights)}
    print(json.dumps({'type': 'review_model', 'model_directory': str(model_path),
                      'assets_sha256': assets, 'transformers': transformers.__version__,
                      'torch': torch.__version__, 'sample_count': len(samples),
                      'normalization': 'NFC, casefold, letters/numbers/combining marks'}), flush=True)
    for sample in samples:
        path = Path(sample['audio'])
        rate, frames, data = read_audio(path)
        pcm = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
        if not np.any(pcm):
            raise ValueError(f'{path}: audio is silent')
        resampled = np.interp(np.arange(round(frames * 16000 / rate)) * rate / 16000,
                              np.arange(frames), pcm).astype(np.float32)
        inputs = processor(resampled, sampling_rate=16000, return_tensors='pt',
                           return_attention_mask=True)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=440,
                                    language=sample['language'], task='transcribe')
        # Whisper can strip EOS from its returned transcription. Conservatively
        # reject a full decoder window instead of treating a short result as
        # truncated solely because its final token is a word.
        if tokens.shape[-1] >= 440:
            raise ValueError(f'{path}: recognition reached its token limit; split the sample')
        recognized = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        print(json.dumps({**sample, 'type': 'speech_review', 'audio_sha256': sha256(path),
                          'audio_seconds': frames / rate, 'recognized': recognized,
                          'character_error_rate': character_error_rate(sample['text'], recognized)},
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, wave.Error) as error:
        sys.exit(str(error))
