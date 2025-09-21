import os
from pathlib import Path

import numpy as np

CONFIG = {
    "sample_rate": 44100,
    "channels": 1,
    "dtype": np.int16,
    "downloads_dir": Path.home() / "Downloads",
    "temp_dir": Path("/tmp"),
    "hyprland_socket": None,  # Will be auto-detected
    "recording_device": None,  # Use default
    "notification_timeout": 3000,  # milliseconds
    # Whisper transcription settings
    "whisper_model": "small",  # Model size: tiny, base, small, medium, large-v3
    "whisper_device": "cpu",  # Device: cpu, cuda, auto
    "whisper_compute_type": "int8",  # Compute type: int8, int16, float16, float32
    "whisper_language": None,  # Language code (None for auto-detection)
    "transcription_timeout": 30,  # Maximum transcription time in seconds
    # Daemon settings
    "daemon_socket": Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    / "speechshift.sock",
    "daemon_pid_file": Path.home() / ".speechshift_daemon.pid",
    "daemon_startup_timeout": 10,  # seconds to wait for daemon startup
    "daemon_lock_file": Path.home() / ".speechshift_daemon.lock",
    "daemon_startup_lock_file": Path.home() / ".speechshift_startup.lock",
    "daemon_shutdown_timeout": 10,  # seconds
    "client_retry_attempts": 3,
    "client_retry_delay": 1.0,  # seconds
}
