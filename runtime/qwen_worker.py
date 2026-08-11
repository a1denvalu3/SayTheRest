#!/usr/bin/env python3
"""Persistent, offline Qwen3-TTS worker for sayIt.

The Rust service owns this process and exchanges one JSON object per line. Model
paths and audio paths are always local; Hugging Face network access is disabled
before importing the inference stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


MODES = {"custom-voice", "voice-design", "voice-clone"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
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


def synthesize(model: Any, mode: str, request: dict[str, Any], torch: Any, soundfile: Any) -> None:
    text = require_text(request, "text")
    output = output_file(request)
    language = optional_text(request, "language") or "Auto"
    seed = request.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or seed < 0 or seed > 0xFFFF_FFFF:
            raise ValueError("seed must be an unsigned 32-bit integer")
        torch.manual_seed(seed)

    if mode == "custom-voice":
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=require_text(request, "speaker"),
            instruct=optional_text(request, "voice_description") or "",
        )
    elif mode == "voice-design":
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=language,
            instruct=require_text(request, "voice_description"),
        )
    else:
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=local_file(request, "reference_audio"),
            ref_text=optional_text(request, "reference_text"),
            x_vector_only_mode=optional_text(request, "reference_text") is None,
        )
    soundfile.write(str(output), wavs[0], sample_rate, subtype="PCM_16")
    if not output.is_file() or output.stat().st_size <= 44:
        raise RuntimeError("inference did not produce a valid WAV payload")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("model must be a local directory")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import soundfile
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=args.device,
        dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            synthesize(model, args.mode, request, torch, soundfile)
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
