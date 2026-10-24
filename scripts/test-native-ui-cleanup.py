#!/usr/bin/env python3
"""Regression for WebView2's reproduced asynchronous profile-cleanup race."""
import errno
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('native_ui_runner',Path(__file__).with_name('test-desktop-ui.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class NativeCleanupTests(unittest.TestCase):
    def test_recreated_cache_entry_is_retried_and_really_removed(self):
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'profile';root.mkdir();(root/'state').write_text('private test state')
            original=shutil.rmtree;attempts=[]
            def remove(path):
                attempts.append(path)
                if len(attempts)==1:
                    (root/'state').unlink()
                    cache=root/'Code Cache/js';cache.mkdir(parents=True);(cache/'late-write').write_bytes(b'test cache')
                    error=OSError(errno.ENOTEMPTY,'WebView2 recreated a cache entry');error.winerror=145
                    raise error
                original(path)
            with patch.object(runner.shutil,'rmtree',side_effect=remove):runner.remove_test_directory(root,windows=True)
            self.assertEqual(len(attempts),2)
            self.assertFalse(root.exists())

    def test_unrelated_io_errors_are_reported_immediately(self):
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'profile';root.mkdir()
            error=OSError(errno.EIO,'unrelated storage error');error.winerror=1117
            with patch.object(runner.shutil,'rmtree',side_effect=error) as remove:
                with self.assertRaises(OSError):runner.remove_test_directory(root,windows=True)
                self.assertEqual(remove.call_count,1)
            self.assertTrue(root.exists())

    def test_persistent_transient_error_has_a_deadline(self):
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'profile';root.mkdir()
            error=OSError(errno.ENOTEMPTY,'still in use');error.winerror=145
            with patch.object(runner.shutil,'rmtree',side_effect=error) as remove:
                with self.assertRaises(OSError):runner.remove_test_directory(root,windows=True,timeout=0)
                self.assertEqual(remove.call_count,1)

    def test_an_already_removed_directory_is_clean(self):
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'already-gone'
            runner.remove_test_directory(root,windows=True)


if __name__=='__main__':unittest.main()
