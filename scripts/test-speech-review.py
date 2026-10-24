#!/usr/bin/env python3
"""Regression checks for the offline speech-review metrics and input limits."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location('speech_review', Path(__file__).with_name('review-speech-audio.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class SpeechReviewTests(unittest.TestCase):
    def test_hindi_vowel_and_nasal_marks_count_as_errors(self):
        self.assertEqual(review.normalize_text('मौसम हैं।'), 'मौसमहैं')
        self.assertGreater(review.character_error_rate('मौसम हैं।', 'मोसम है।'), 0)

    def test_canonical_equivalence_and_punctuation(self):
        self.assertEqual(review.character_error_rate('Café, CLEAR!', 'cafe\u0301 clear'), 0)
        self.assertEqual(review.character_error_rate('abc', 'ac'), 1 / 3)
        self.assertEqual(review.character_error_rate('abc', 'abxc'), 1 / 3)
        with self.assertRaises(ValueError):
            review.character_error_rate('... ', '')

    def test_no_unrequested_simplification_or_numeric_equivalence(self):
        self.assertGreater(review.character_error_rate('安静', '安靜'), 0)
        self.assertGreater(review.character_error_rate('35', 'thirty five'), 0)

    def test_long_stereo_and_truncated_wavs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'sample.wav'
            for channels, frames in [(1, 8000), (2, 8000), (1, 240001)]:
                with wave.open(str(path), 'wb') as audio:
                    audio.setparams((channels, 2, 8000, 0, 'NONE', 'not compressed'))
                    audio.writeframes(b'\1\0' * frames * channels)
                if channels == 1 and frames == 8000:
                    self.assertEqual(review.read_audio(path)[:2], (8000, 8000))
                else:
                    with self.assertRaises(ValueError):
                        review.read_audio(path)
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\1\0' * 8000)
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                review.read_audio(path)


if __name__ == '__main__':
    unittest.main()
