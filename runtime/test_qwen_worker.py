import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "qwen_worker", Path(__file__).with_name("qwen_worker.py")
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class WorkerValidationTests(unittest.TestCase):
    def test_resource_threads_preserve_capacity_for_the_desktop(self):
        self.assertEqual(WORKER.resource_thread_count(1), 1)
        self.assertEqual(WORKER.resource_thread_count(2), 1)
        self.assertEqual(WORKER.resource_thread_count(4), 2)
        self.assertEqual(WORKER.resource_thread_count(64), 2)

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


if __name__ == "__main__":
    unittest.main()
