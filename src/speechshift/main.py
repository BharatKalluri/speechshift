#!/usr/bin/env python3
"""
SpeechShift POC for Arch Linux + Hyprland
A speech-to-text application that records audio and transcribes it using faster-whisper

Requirements:
- Arch Linux with Hyprland
- PipeWire audio system
- mako notifications
- wl-clipboard and wtype for Wayland text input
- faster-whisper for transcription

Usage:
    speechshift --daemon    # Run as background daemon
    speechshift --toggle    # Toggle recording (start/stop)
    speechshift --test      # Test all components including transcription
"""

import os
import sys
from pathlib import Path
import argparse

from speechshift.core.speechshift_daemon import SimpleDaemon, SimpleDaemonClient

# Audio recording imports
try:
    import sounddevice as sd
    import numpy as np
    import wave

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print(
        "Warning: sounddevice not available. Install dependencies manually."
    )

# Whisper transcription imports
try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print(
        "Warning: faster-whisper not available. Install dependencies manually."
    )

# Configuration
CONFIG = {
    "sample_rate": 44100,
    "channels": 1,
    "dtype": np.int16 if AUDIO_AVAILABLE else None,
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
    "daemon_socket": Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "speechshift.sock",
    "daemon_pid_file": Path.home() / ".speechshift_daemon.pid",
    "daemon_startup_timeout": 10,  # seconds to wait for daemon startup
    "daemon_lock_file": Path.home() / ".speechshift_daemon.lock",
    "daemon_startup_lock_file": Path.home() / ".speechshift_startup.lock",
    "daemon_shutdown_timeout": 10,  # seconds
    "client_retry_attempts": 3,
    "client_retry_delay": 1.0,  # seconds
}


# Persistent state management


# Global state
class RecordingState:
    def __init__(self):
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = []
        self.current_file = None
        self.start_time = None
        # Simplified state - only one flag for transcription
        self.is_transcribing = False


recording_state = RecordingState()




def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SpeechShift POC for Arch Linux + Hyprland"
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run as background daemon"
    )
    parser.add_argument("--toggle", action="store_true", help="Toggle recording")
    parser.add_argument("--test", action="store_true", help="Test system components")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown daemon")

    args = parser.parse_args()

    if args.test:
        print("Testing SpeechShift components...")

        # Test audio
        if AUDIO_AVAILABLE:
            try:
                devices = sd.query_devices()
                print(f"Audio devices available: {len(devices)}")
                print(f"Default input device: {sd.default.device[0]}")
            except Exception as e:
                print(f"Audio test failed: {e}")
        else:
            print("Audio not available - install dependencies manually")

        # Test Whisper transcription
        if WHISPER_AVAILABLE:
            try:
                from speechshift.core.whisper_transcriber import WhisperTranscriber
                transcriber = WhisperTranscriber()
                if transcriber.is_available():
                    print(
                        f"Whisper transcription available: model={CONFIG['whisper_model']}"
                    )
                else:
                    print("Whisper model loading failed")
            except Exception as e:
                print(f"Whisper test failed: {e}")
        else:
            print("Whisper not available - install dependencies manually")

        # Test Wayland tools
        import subprocess
        def check_command(cmd):
            try:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        
        wayland_ok = all([check_command("wl-copy"), check_command("wtype")])
        print(f"Wayland tools available: {wayland_ok}")

        return 0

    if args.daemon:
        # Start daemon directly
        daemon = SimpleDaemon()
        return 0 if daemon.start_daemon() else 1

    # For all other commands, use daemon client
    client = SimpleDaemonClient()

    if args.status:
        response = client.send_command("status")
        if response:
            if response.startswith("ERROR"):
                print(f"Daemon error: {response}")
                return 1
            else:
                print(f"Daemon status: {response}")
                return 0
        else:
            print("Failed to communicate with daemon")
            return 1

    elif args.shutdown:
        response = client.send_command("shutdown")
        if response:
            print(response)
            return 0 if response.startswith("OK") else 1
        else:
            print("Failed to shutdown daemon")
            return 1

    elif args.toggle:
        response = client.send_command("toggle")
        if response:
            print(response)
            return 0 if response.startswith("OK") else 1
        else:
            print("Failed to communicate with daemon")
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
