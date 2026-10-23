import contextlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import wave
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SPEC = importlib.util.spec_from_file_location(
    "qwen_worker", Path(__file__).with_name("qwen_worker.py")
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class WaveSamples(list):
    pass


class WaveWriter:
    def __init__(self, path, *, mode, samplerate, channels, subtype):
        self.samplerate = samplerate
        self.writer = wave.open(path, "wb")
        self.writer.setparams((channels, 2, samplerate, 0, "NONE", "NONE"))

    def write(self, samples):
        self.writer.writeframes(b"\0\0" * len(samples))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.writer.close()


class WorkerValidationTests(unittest.TestCase):
    def test_automatic_device_selection_uses_available_hardware_and_cpu_fallback(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
            xpu=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        self.assertEqual(WORKER.inference_device(torch, "auto"), "cuda:0")
        self.assertEqual(WORKER.inference_dtype(torch, "cuda:0", "auto"), "bfloat16")
        torch.cuda.is_bf16_supported = lambda: False
        self.assertEqual(WORKER.inference_dtype(torch, "cuda:0", "auto"), "float16")
        torch.cuda.is_available = lambda: False
        self.assertEqual(WORKER.inference_device(torch, "auto"), "xpu:0")
        torch.xpu.is_available = lambda: False
        self.assertEqual(WORKER.inference_device(torch, "auto"), "mps")
        torch.backends.mps.is_available = lambda: False
        self.assertEqual(WORKER.inference_device(torch, "auto"), "cpu")
        self.assertEqual(WORKER.inference_dtype(torch, "cpu", "auto"), "float32")
        self.assertEqual(WORKER.inference_device(torch, "cpu"), "cpu")
        self.assertEqual(WORKER.inference_dtype(torch, "cuda:0", "float32"), "float32")

    def test_resource_threads_preserve_capacity_for_the_desktop(self):
        self.assertEqual(WORKER.resource_thread_count(1), 1)
        self.assertEqual(WORKER.resource_thread_count(2), 1)
        self.assertEqual(WORKER.resource_thread_count(4), 2)
        self.assertEqual(WORKER.resource_thread_count(8), 4)
        self.assertEqual(WORKER.resource_thread_count(12), 6)
        self.assertEqual(WORKER.resource_thread_count(64), 8)

    def test_required_text_rejects_blank_and_non_string_values(self):
        for request in ({}, {"text": "  "}, {"text": 7}):
            with self.assertRaises(ValueError):
                WORKER.require_text(request, "text")

    def test_reference_audio_must_be_an_existing_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.wav"
            audio.write_bytes(b"RIFF")
            self.assertEqual(
                WORKER.local_file({"reference_audio": str(audio)}, "reference_audio"),
                str(audio.resolve()),
            )
            with self.assertRaises(ValueError):
                WORKER.local_file(
                    {"reference_audio": "https://example.com/voice.wav"},
                    "reference_audio",
                )

    def test_output_parent_is_created_without_requiring_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "speech.wav"
            self.assertEqual(WORKER.output_file({"output": str(output)}), output)
            self.assertTrue(output.parent.is_dir())

    def test_damaged_model_config_is_rejected_before_importing_inference_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({"engine": "qwen3-tts"}))
            # Isolated system Python needs none of the heavyweight dependencies:
            # the damaged config must fail before their imports or allocations.
            result = subprocess.run(
                [sys.executable, "-I", SPEC.origin,
                 "--model", directory, "--mode", "voice-clone"],
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("configuration is missing or damaged", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_pinned_architecture_is_a_valid_tts_config(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "crates/service/data/qwen3-tts-06b-base.config.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_bytes(config.read_bytes())
            WORKER.validate_model_directory(Path(directory))

    def test_text_chunks_preserve_content_and_bound_unbroken_and_multilingual_text(self):
        for text in ("Hello world. " * 100, "字" * 1000, "x" * 1000, "\nOne.\nTwo!\n" * 100):
            with self.subTest(text=text[:20]):
                chunks = list(WORKER.text_chunks(text))
                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(
                    0 < len(chunk) <= WORKER.MAX_CHUNK_CHARACTERS for chunk in chunks
                ))
                self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))


class WorkerSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "speech.wav"
        self.reference = self.directory / "voice.wav"
        self.reference.write_bytes(b"RIFF")
        self.torch = SimpleNamespace(inference_mode=Mock(side_effect=contextlib.nullcontext))
        self.soundfile = SimpleNamespace(
            SoundFile=Mock(side_effect=WaveWriter),
            info=Mock(return_value=SimpleNamespace(duration=14.5)),
        )
        self.model = Mock()
        self.previous_samples = None

    def generate(self, **options):
        # Keep only a weak reference: every chunk must be released before the
        # next inference call, instead of accumulating all waveforms in a list.
        if self.previous_samples is not None:
            self.assertIsNone(self.previous_samples())
        samples = WaveSamples([0.0] * 24)
        self.previous_samples = weakref.ref(samples)
        return [samples], 24000

    def request(self, **overrides):
        return {
            "text": "A short sentence. " * 30,
            "output": str(self.output),
            "reference_audio": str(self.reference),
            "speaker": "Aiden",
            "voice_description": "Calm and clear",
            **overrides,
        }

    def test_all_modes_stream_bounded_chunks_to_one_wav(self):
        for mode, method in (
            ("voice-clone", "generate_voice_clone"),
            ("voice-design", "generate_voice_design"),
            ("custom-voice", "generate_custom_voice"),
        ):
            with self.subTest(mode=mode):
                generate = getattr(self.model, method)
                generate.side_effect = self.generate
                request = self.request()
                WORKER.synthesize(self.model, mode, request, self.torch, self.soundfile)
                self.assertGreater(generate.call_count, 1)
                for call in generate.call_args_list:
                    self.assertLessEqual(len(call.kwargs["text"]), WORKER.MAX_CHUNK_CHARACTERS)
                    self.assertEqual(call.kwargs["max_new_tokens"], WORKER.MAX_NEW_TOKENS)
                with wave.open(str(self.output)) as reader:
                    self.assertEqual(reader.getnframes(), generate.call_count * 24)
                    self.assertEqual(reader.getframerate(), 24000)
        self.assertEqual(self.model.create_voice_clone_prompt.call_count, 1)
        prompt = self.model.create_voice_clone_prompt.return_value
        for call in self.model.generate_voice_clone.call_args_list:
            self.assertIs(call.kwargs["voice_clone_prompt"], prompt)
        self.assertEqual(self.torch.inference_mode.call_count, 3)

    def test_pcm_is_emitted_before_publishing_the_complete_file(self):
        self.model.generate_voice_clone.side_effect = self.generate
        events = []
        def emit(event):
            self.assertFalse(self.output.exists())
            self.assertEqual(event["audio"]["sample_rate"], 24000)
            events.append(event)
        WORKER.synthesize(self.model, "voice-clone", self.request(stream=True),
                          self.torch, self.soundfile, emit)
        self.assertGreater(len(events), 1)
        with wave.open(str(self.output)) as wav:
            self.assertEqual(sum(len(e["audio"]["samples"]) for e in events), wav.getnframes())

    def test_failed_chunk_does_not_publish_partial_audio_or_replace_existing_output(self):
        self.output.write_bytes(b"existing audio")
        self.model.generate_voice_clone.side_effect = [
            self.generate(), RuntimeError("inference failed")
        ]
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            WORKER.synthesize(self.model, "voice-clone", self.request(), self.torch, self.soundfile)
        self.assertEqual(self.output.read_bytes(), b"existing audio")
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()), ["speech.wav", "voice.wav"]
        )

    def test_long_reference_is_rejected_before_prompt_encoding(self):
        self.soundfile.info.return_value.duration = 3600
        with self.assertRaisesRegex(ValueError, "at most 45 seconds"):
            WORKER.synthesize(self.model, "voice-clone", self.request(), self.torch, self.soundfile)
        self.model.create_voice_clone_prompt.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_longer_reference_is_accepted_with_its_complete_transcript(self):
        self.soundfile.info.return_value.duration = 44.5
        self.model.generate_voice_clone.side_effect = self.generate
        WORKER.synthesize(
            self.model, "voice-clone", self.request(reference_text="A longer, varied reference passage."),
            self.torch, self.soundfile,
        )
        self.model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=str(self.reference), ref_text="A longer, varied reference passage.", x_vector_only_mode=False,
        )

    def test_reference_transcript_enables_icl_in_the_reused_prompt(self):
        self.model.generate_voice_clone.side_effect = self.generate
        WORKER.synthesize(
            self.model, "voice-clone", self.request(reference_text="Reference transcript"),
            self.torch, self.soundfile,
        )
        self.model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=str(self.reference), ref_text="Reference transcript", x_vector_only_mode=False,
        )

    def test_legacy_profile_without_transcript_uses_embedding_only_after_icl_request(self):
        self.model.generate_voice_clone.side_effect = self.generate
        WORKER.synthesize(
            self.model, "voice-clone", self.request(reference_text="Exact reference words."),
            self.torch, self.soundfile,
        )
        WORKER.synthesize(self.model, "voice-clone", self.request(), self.torch, self.soundfile)
        self.assertEqual(self.model.create_voice_clone_prompt.call_count, 2)
        self.assertEqual(self.model.create_voice_clone_prompt.call_args.kwargs, {
            "ref_audio": str(self.reference), "ref_text": None, "x_vector_only_mode": True,
        })


if __name__ == "__main__":
    unittest.main()
