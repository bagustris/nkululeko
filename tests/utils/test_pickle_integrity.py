# test_pickle_integrity.py - unit tests for nkululeko/utils/pickle_integrity.py
import os
import pickle
import tempfile
import unittest

from nkululeko.utils.pickle_integrity import (
    _checksum_path,
    save_checksum,
    verify_checksum,
)


class TestPickleIntegrity(unittest.TestCase):
    def _create_pickle(self, tmpdir, obj, name="test.pkl"):
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path

    def test_save_and_verify_checksum_success(self):
        """Checksum verification passes for unmodified file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_pickle(tmpdir, {"key": "value"})
            save_checksum(path)
            # Should not raise
            verify_checksum(path)
            # Checksum file should exist
            self.assertTrue(os.path.isfile(_checksum_path(path)))

    def test_verify_checksum_tampered_file(self):
        """Checksum verification fails for tampered file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_pickle(tmpdir, {"key": "value"})
            save_checksum(path)
            # Tamper with the pickle file
            with open(path, "ab") as f:
                f.write(b"tampered")
            with self.assertRaises(ValueError) as ctx:
                verify_checksum(path)
            self.assertIn("Checksum mismatch", str(ctx.exception))

    def test_verify_checksum_no_checksum_file(self):
        """Verification warns but does not raise when no checksum file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_pickle(tmpdir, {"key": "value"})
            # No checksum file saved - should not raise (backward compat)
            verify_checksum(path)

    def test_checksum_path(self):
        """_checksum_path appends .sha256 to the file path."""
        self.assertEqual(_checksum_path("/path/to/file.pkl"), "/path/to/file.pkl.sha256")

    def test_save_checksum_creates_correct_hash(self):
        """Saved checksum matches manual SHA256 computation."""
        import hashlib

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_pickle(tmpdir, [1, 2, 3])
            save_checksum(path)
            # Manually compute hash
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                sha256.update(f.read())
            expected = sha256.hexdigest()
            with open(_checksum_path(path), "r") as f:
                stored = f.read().strip()
            self.assertEqual(stored, expected)


if __name__ == "__main__":
    unittest.main()
