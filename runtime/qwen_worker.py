#!/usr/bin/env python3
"""Persistent, offline Qwen3-TTS worker for sayIt.

The Rust service owns this process and exchanges one JSON object per line. Model
paths and audio paths are always local; Hugging Face network access is disabled
before importing the inference stack.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any


MODES = {"custom-voice", "voice-design", "voice-clone"}
MAX_CPU_THREADS = 8
# The shipped generation_config permits 8192 codec frames (over ten minutes).
# Bound each generation and write chunks to disk so text length does not set the
# size of the KV cache, retained hidden states, or decoded waveform in memory.
MAX_CHUNK_CHARACTERS = 160
MAX_NEW_TOKENS = 512
MAX_REFERENCE_SECONDS = 45


def resource_thread_count(cpu_count: int | None = None) -> int:
    """Use half the CPUs while leaving at least two available to the desktop."""
    count = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(MAX_CPU_THREADS, count // 2, max(1, count - 2)))


def configure_resource_limits() -> int:
    threads = resource_thread_count()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # Inference is background work. Lowering its scheduler priority keeps input,
    # audio, and the desktop responsive when the CPU is under sustained load.
    try:
        if os.name == "nt":
            import ctypes

            below_normal_priority_class = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                below_normal_priority_class,
            )
        else:
            os.nice(10)
    except (AttributeError, OSError):
        pass
    return threads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    return parser.parse_args()


def require_text(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def optional_text(request: dict[str, Any], name: str) -> str | None:
    value = request.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    return value or None


def local_file(request: dict[str, Any], name: str) -> str:
    try:
        value = Path(require_text(request, name)).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is not a local file") from error
    if not value.is_file():
        raise ValueError(f"{name} is not a local file")
    return str(value)


def output_file(request: dict[str, Any]) -> Path:
    value = Path(require_text(request, "output")).expanduser().resolve(strict=False)
    value.parent.mkdir(parents=True, exist_ok=True)
    return value


def validate_model_directory(model_path: Path) -> None:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("model_type") != "qwen3_tts"
        or not isinstance(config.get("talker_config"), dict)
        or not config["talker_config"]
    ):
        # AutoConfig otherwise guesses from the directory name and can select a
        # much larger generic Qwen3 language model. Reject before importing torch.
        raise ValueError(
            "Qwen model configuration is missing or damaged; update the model in sayIt "
            "to restore its TTS architecture"
        )


def text_chunks(text: str) -> Iterator[str]:
    remaining = text.strip()
    while remaining:
        if len(remaining) <= MAX_CHUNK_CHARACTERS:
            yield remaining
            return
        prefix = remaining[:MAX_CHUNK_CHARACTERS]
        boundaries = list(re.finditer(r"[.!?;](?:\s|$)|[。！？；]|\n", prefix))
        end = boundaries[-1].end() if boundaries else 0
        if not end:
            spaces = list(re.finditer(r"\s+", prefix))
            end = spaces[-1].end() if spaces else MAX_CHUNK_CHARACTERS
        chunk = remaining[:end].strip()
        if chunk:
            yield chunk
        remaining = remaining[end:].lstrip()


def synthesize(model: Any, mode: str, request: dict[str, Any], torch: Any, soundfile: Any, emit=None) -> None:
    text = require_text(request, "text")
    output = output_file(request)
    language = optional_text(request, "language") or "Auto"
    seed = request.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or seed < 0 or seed > 0xFFFF_FFFF:
            raise ValueError("seed must be an unsigned 32-bit integer")
        torch.manual_seed(seed)

    with torch.inference_mode(), tempfile.TemporaryDirectory(
        prefix=".sayit-qwen-", dir=output.parent
    ) as temporary, contextlib.ExitStack() as stack:
        options: dict[str, Any] = {"language": language, "max_new_tokens": MAX_NEW_TOKENS}
        if mode == "custom-voice":
            generate = model.generate_custom_voice
            options["speaker"] = require_text(request, "speaker")
            options["instruct"] = optional_text(request, "voice_description") or ""
        elif mode == "voice-design":
            generate = model.generate_voice_design
            options["instruct"] = require_text(request, "voice_description")
        else:
            generate = model.generate_voice_clone
            reference = local_file(request, "reference_audio")
            info = soundfile.info(reference)
            if not 0 < info.duration <= MAX_REFERENCE_SECONDS:
                raise ValueError(f"reference audio must contain at most {MAX_REFERENCE_SECONDS} seconds of speech")
            reference_text = optional_text(request, "reference_text")
            # Reuse just this request's prompt, without retaining tensors from
            # previous requests or re-encoding the reference for every chunk.
            options["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=reference,
                ref_text=reference_text,
                x_vector_only_mode=reference_text is None,
            )

        pending = Path(temporary) / "speech.wav"
        writer = None
        for chunk in text_chunks(text):
            wavs, sample_rate = generate(text=chunk, **options)
            if len(wavs) != 1 or len(wavs[0]) == 0:
                raise RuntimeError("inference did not produce audio")
            if writer is None:
                writer = stack.enter_context(
                    soundfile.SoundFile(
                        str(pending), mode="w", samplerate=sample_rate,
                        channels=1, subtype="PCM_16",
                    )
                )
            elif sample_rate != writer.samplerate:
                raise RuntimeError("inference changed the sample rate between chunks")
            writer.write(wavs[0])
            if request.get("stream") and emit is not None:
                for start in range(0, len(wavs[0]), 4096):
                    emit({"audio": {"sample_rate": sample_rate,
                                    "samples": [float(value) for value in wavs[0][start:start + 4096]]}})
            del wavs
        stack.close()
        if not pending.is_file() or pending.stat().st_size <= 44:
            raise RuntimeError("inference did not produce a valid WAV payload")
        pending.replace(output)


def inference_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def inference_dtype(torch: Any, device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if device.startswith("cuda"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("model must be a local directory")
    validate_model_directory(model_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    threads = configure_resource_limits()

    # Libraries occasionally print to stdout; reserve it for the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import soundfile
        import torch
        from qwen_tts import Qwen3TTSModel

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

        device = inference_device(torch, args.device)
        dtype_name = inference_dtype(torch, device, args.dtype)
        dtype = getattr(torch, dtype_name)
        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    print(json.dumps({"ready": True, "device": device, "dtype": dtype_name}), flush=True)

    protocol = sys.stdout
    def emit(event):
        print(json.dumps(event, allow_nan=False), file=protocol, flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            with contextlib.redirect_stdout(sys.stderr):
                synthesize(model, args.mode, request, torch, soundfile, emit)
            print(json.dumps({"ok": True}), flush=True)
        except Exception as error:  # Keep the loaded model alive after a bad request.
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(error),
                        "trace": traceback.format_exc(limit=3),
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
