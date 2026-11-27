#!/usr/bin/env python3
"""Persistent, offline Qwen3-TTS worker for sayIt.

The Rust service owns this process and exchanges one JSON object per line. Model
paths and audio paths are always local; Hugging Face network access is disabled
before importing the inference stack.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import traceback
import weakref
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

# Bounded 1-entry cache per loaded model instance for voice-clone prompts so
# repeated requests with the same reference WAV + transcript avoid re-encoding
# (~1.5s saved per block on CPU). WeakKeyDictionary automatically frees entries
# when a model instance is released.
_VOICE_CLONE_PROMPT_CACHE: weakref.WeakKeyDictionary[Any, tuple[Any, Any]] = (
    weakref.WeakKeyDictionary()
)


def _linux_preferred_physical_cpus(limit: int) -> list[int]:
    """Return up to `limit` primary logical CPUs ordered fastest-first on Linux."""
    if not sys.platform.startswith("linux") or limit <= 0:
        return []
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.is_dir():
        return []
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        allowed = None
    seen: dict[tuple[int, int, int], tuple[int, int]] = {}
    for entry in cpu_root.iterdir():
        name = entry.name
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        cpu_id = int(name[3:])
        if allowed is not None and cpu_id not in allowed:
            continue
        try:
            online_path = entry / "online"
            if online_path.is_file() and online_path.read_text().strip() == "0":
                continue
            pkg = int((entry / "topology/physical_package_id").read_text().strip())
        except OSError:
            pkg = 0
        try:
            core = int((entry / "topology/core_id").read_text().strip())
        except OSError:
            core = cpu_id
        try:
            sib = (entry / "topology/thread_siblings_list").read_text().strip()
            primary = int(sib.split(",")[0].split("-")[0])
        except (OSError, ValueError):
            primary = cpu_id
        score = 0
        for rel in ("cpufreq/cpuinfo_max_freq", "cpu_capacity"):
            try:
                score = int((entry / rel).read_text().strip())
                break
            except (OSError, ValueError):
                pass
        key = (pkg, core, primary)
        prev = seen.get(key)
        if prev is None or cpu_id < prev[0] or score > prev[1]:
            seen[key] = (min(cpu_id, prev[0]) if prev else cpu_id, max(score, prev[1]) if prev else score)
    ranked = sorted(seen.values(), key=lambda item: (-item[1], item[0]))
    return [cpu_id for cpu_id, _ in ranked[:limit]]


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
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    preferred_cpus = _linux_preferred_physical_cpus(threads)
    if preferred_cpus:
        try:
            os.sched_setaffinity(0, set(preferred_cpus))
        except (AttributeError, OSError):
            pass

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


def generation_options(request: dict[str, Any]) -> dict[str, Any]:
    limit = request.get("max_new_tokens", MAX_NEW_TOKENS)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 2048:
        raise ValueError("max_new_tokens must be an integer between 1 and 2048")
    options: dict[str, Any] = {"max_new_tokens": limit}
    for key, low, high in [("temperature", 0.2, 1.2), ("top_p", 0.5, 1.0), ("repetition_penalty", 0.9, 1.5)]:
        if key not in request:
            continue
        value = request[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        options[key] = value
    if "top_k" in request:
        value = request["top_k"]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError("top_k must be an integer between 0 and 100")
        options["top_k"] = value
    return options


def synthesize(model: Any, mode: str, request: dict[str, Any], torch: Any, soundfile: Any, emit=None) -> None:
    text = require_text(request, "text")
    output = output_file(request)
    language = optional_text(request, "language") or "Auto"
    seed = request.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 0xFFFF_FFFF:
            raise ValueError("seed must be an unsigned 32-bit integer")
        torch.manual_seed(seed)

    with torch.inference_mode(), tempfile.TemporaryDirectory(
        prefix=".sayit-qwen-", dir=output.parent
    ) as temporary, contextlib.ExitStack() as stack:
        options: dict[str, Any] = {"language": language, **generation_options(request)}
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
            x_vector_only = reference_text is None
            ref_digest = hashlib.sha256(Path(reference).read_bytes()).digest()
            cache_key = (ref_digest, reference_text, x_vector_only)
            cached_prompt = None
            try:
                entry = _VOICE_CLONE_PROMPT_CACHE.get(model)
                if entry is not None and entry[0] == cache_key:
                    cached_prompt = entry[1]
            except TypeError:
                pass
            if cached_prompt is None:
                cached_prompt = model.create_voice_clone_prompt(
                    ref_audio=reference,
                    ref_text=reference_text,
                    x_vector_only_mode=x_vector_only,
                )
                try:
                    _VOICE_CLONE_PROMPT_CACHE[model] = (cache_key, cached_prompt)
                except TypeError:
                    pass
            options["voice_clone_prompt"] = cached_prompt

        pending = Path(temporary) / "speech.wav"
        writer = None
        use_b64 = request.get("stream_format") == "pcm_f32le_b64"
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
                    piece = wavs[0][start:start + 4096]
                    if use_b64:
                        if hasattr(piece, "astype"):
                            raw = piece.astype("<f4", copy=False).tobytes()
                        else:
                            import struct
                            raw = struct.pack(f"<{len(piece)}f", *(float(v) for v in piece))
                        emit({"audio": {"sample_rate": sample_rate,
                                        "pcm_f32le_b64": base64.b64encode(raw).decode("ascii")}})
                    else:
                        emit({"audio": {"sample_rate": sample_rate,
                                        "samples": [float(value) for value in piece]}})
                    del piece
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
