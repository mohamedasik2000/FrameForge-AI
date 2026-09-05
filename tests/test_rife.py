"""
Unit tests for RIFEInterpolator — checksum verification, download logic, GPU validation.

These tests do NOT require a CUDA GPU or the actual model weights.
They test the error paths, validation logic, and configuration propagation.
"""

import hashlib
import os
import struct
import tempfile
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from frameforge.interpolation.rife import (
    ChecksumError,
    RIFEInterpolator,
    _DEFAULT_SHA256,
    _DEFAULT_MODEL_FILENAME,
    _DEFAULT_DOWNLOAD_URL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_fake_checkpoint(path: str, content: bytes = b"fake model data"):
    with open(path, "wb") as f:
        f.write(content)
    return content


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

class TestChecksumVerification:
    def test_verify_correct_checksum(self, tmp_path):
        content = b"model weights content"
        p = tmp_path / "model.pkl"
        p.write_bytes(content)
        interp = RIFEInterpolator.__new__(RIFEInterpolator)
        # Should not raise
        interp._verify_checksum(str(p), sha256_of(content))

    def test_verify_wrong_checksum_raises(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_bytes(b"correct content")
        interp = RIFEInterpolator.__new__(RIFEInterpolator)
        with pytest.raises(ChecksumError, match="checksum mismatch"):
            interp._verify_checksum(str(p), "0" * 64)

    def test_verify_empty_file_raises(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_bytes(b"")
        interp = RIFEInterpolator.__new__(RIFEInterpolator)
        with pytest.raises(ChecksumError):
            interp._verify_checksum(str(p), "a" * 64)

    def test_verify_corrupted_cache_raises(self, tmp_path):
        """A cached model with wrong bytes must fail verification."""
        content = b"corrupted garbage"
        p = tmp_path / _DEFAULT_MODEL_FILENAME
        p.write_bytes(content)
        interp = RIFEInterpolator(gpu_id=0)

        with pytest.raises(ChecksumError):
            interp._verify_checksum(str(p), _DEFAULT_SHA256)

    def test_compute_sha256(self, tmp_path):
        content = b"test content"
        p = tmp_path / "f.bin"
        p.write_bytes(content)
        interp = RIFEInterpolator.__new__(RIFEInterpolator)
        result = interp._compute_sha256(str(p))
        assert result == hashlib.sha256(content).hexdigest().lower()


# ---------------------------------------------------------------------------
# No silent bypass
# ---------------------------------------------------------------------------

class TestNoBypass:
    def test_no_bypass_when_expected_hash_provided(self, tmp_path):
        """Providing expected_sha256=None does NOT skip verification for default model."""
        content = b"wrong content"
        model_dir = str(tmp_path)
        p = tmp_path / _DEFAULT_MODEL_FILENAME
        p.write_bytes(content)

        # RIFEInterpolator with default sha256=None still uses _DEFAULT_SHA256
        interp = RIFEInterpolator(expected_sha256=None)

        # Patching torch to avoid CUDA requirement
        with patch("frameforge.interpolation.rife.Model"):
            with pytest.raises(ChecksumError):
                interp.load_model(model_dir=model_dir, model_filename=_DEFAULT_MODEL_FILENAME)


# ---------------------------------------------------------------------------
# GPU ID validation
# ---------------------------------------------------------------------------

class TestGPUValidation:
    def test_negative_gpu_id_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RIFEInterpolator(gpu_id=-1)

    def test_string_gpu_id_raises(self):
        with pytest.raises((ValueError, TypeError)):
            RIFEInterpolator(gpu_id="cuda:0")  # type: ignore


# ---------------------------------------------------------------------------
# Download failure handling
# ---------------------------------------------------------------------------

class TestDownloadFailure:
    def test_failed_download_raises_runtime_error(self, tmp_path):
        interp = RIFEInterpolator()

        with patch("frameforge.interpolation.rife.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Network unreachable")
            with pytest.raises(RuntimeError, match="Failed to download"):
                interp.load_model(model_dir=str(tmp_path))

    def test_http_error_raises(self, tmp_path):
        interp = RIFEInterpolator()

        with patch("frameforge.interpolation.rife.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
            mock_get.return_value = mock_resp
            with pytest.raises(RuntimeError, match="Failed to download"):
                interp.load_model(model_dir=str(tmp_path))

    def test_tmp_file_cleaned_up_on_failure(self, tmp_path):
        interp = RIFEInterpolator()

        with patch("frameforge.interpolation.rife.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("fail")
            try:
                interp.load_model(model_dir=str(tmp_path))
            except RuntimeError:
                pass
            # No .tmp files should remain
            tmp_files = list(tmp_path.glob("*.tmp"))
            assert len(tmp_files) == 0, f"Orphaned .tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Custom model without download
# ---------------------------------------------------------------------------

class TestCustomModel:
    def test_custom_model_not_found_raises(self, tmp_path):
        interp = RIFEInterpolator(expected_sha256="a" * 64)
        with pytest.raises(FileNotFoundError, match="Custom model not found"):
            interp.load_model(
                model_dir=str(tmp_path),
                model_filename="custom_model.pkl",
            )

    def test_custom_model_correct_hash_loads(self, tmp_path):
        content = b"fake custom model"
        expected = sha256_of(content)
        p = tmp_path / "custom.pkl"
        p.write_bytes(content)

        interp = RIFEInterpolator(expected_sha256=expected)
        # Should pass verification but fail on actual model loading (no CUDA)
        with patch("frameforge.interpolation.rife.Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            # Mock torch.load inside the Model class
            interp.load_model(model_dir=str(tmp_path), model_filename="custom.pkl")
            # Model should have been constructed
            MockModel.assert_called_once()


# ---------------------------------------------------------------------------
# Interpolate precondition
# ---------------------------------------------------------------------------

class TestInterpolatePreconditions:
    def test_interpolate_without_model_raises(self):
        interp = RIFEInterpolator()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            interp.interpolate(img, img, 0.5)

    def test_invalid_timestep_raises(self):
        interp = RIFEInterpolator()
        interp.model = MagicMock()  # pretend model is loaded
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="timestep"):
            interp.interpolate(img, img, 1.5)

    def test_negative_timestep_raises(self):
        interp = RIFEInterpolator()
        interp.model = MagicMock()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="timestep"):
            interp.interpolate(img, img, -0.1)


# ---------------------------------------------------------------------------
# Configuration propagation
# ---------------------------------------------------------------------------

class TestConfigPropagation:
    def test_default_fp16_is_false(self):
        interp = RIFEInterpolator()
        assert interp.fp16 is False

    def test_scale_propagated(self):
        interp = RIFEInterpolator(scale=0.5)
        assert interp.scale == 0.5

    def test_gpu_id_propagated(self):
        interp = RIFEInterpolator(gpu_id=0)
        assert interp.gpu_id == 0

    def test_device_uses_gpu_id(self):
        import torch
        interp = RIFEInterpolator(gpu_id=0)
        assert str(interp.device) == "cuda:0"
