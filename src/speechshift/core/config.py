import os
import json
from pathlib import Path

import numpy as np

BASE_CONFIG = {
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
    "transcription_timeout": 120,  # Maximum transcription time in seconds
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


def _get_config_dir():
    """Get XDG-compliant config directory"""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        config_base = Path(xdg_config)
    else:
        config_base = Path.home() / ".config"
    
    config_dir = config_base / "speechshift"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _create_default_config(config_file):
    """Create default config file if it doesn't exist"""
    default_config = {
        "whisper": {
            "model": "small",
            "language": None,
            "device": "cpu",
            "compute_type": "int8"
        },
        "audio": {
            "recording_device": None,
            "notification_timeout": 3000
        }
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
    except Exception:
        pass  # Silently fail if can't create config


def _load_user_config():
    """Load user configuration from JSON file"""
    config_dir = _get_config_dir()
    config_file = config_dir / "config.json"
    
    # Create default config if it doesn't exist
    if not config_file.exists():
        _create_default_config(config_file)
    
    # Try to load user config
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            return user_config
    except Exception:
        return {}  # Return empty dict if can't load


def _merge_configs():
    """Merge user config with base config"""
    config = BASE_CONFIG.copy()
    user_config = _load_user_config()
    
    # Override whisper settings if present
    if "whisper" in user_config:
        whisper_config = user_config["whisper"]
        if "model" in whisper_config:
            config["whisper_model"] = whisper_config["model"]
        if "language" in whisper_config:
            config["whisper_language"] = whisper_config["language"]
        if "device" in whisper_config:
            config["whisper_device"] = whisper_config["device"]
        if "compute_type" in whisper_config:
            config["whisper_compute_type"] = whisper_config["compute_type"]
    
    # Override audio settings if present
    if "audio" in user_config:
        audio_config = user_config["audio"]
        if "recording_device" in audio_config:
            config["recording_device"] = audio_config["recording_device"]
        if "notification_timeout" in audio_config:
            config["notification_timeout"] = audio_config["notification_timeout"]
    
    return config


# Load configuration with user overrides
CONFIG = _merge_configs()
