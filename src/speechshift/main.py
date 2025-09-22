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

import sys
import argparse
import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel

from speechshift.core.config import CONFIG
from speechshift.core.speechshift_daemon import SpeechshiftDaemon, SimpleDaemonClient


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
        try:
            devices = sd.query_devices()
            print(f"Audio devices available: {len(devices)}")
            print(f"Default input device: {sd.default.device[0]}")
        except Exception as e:
            print(f"Audio test failed: {e}")

        # Test Whisper transcription
        try:
            from speechshift.core.audio_transcriber import AudioTranscriber

            transcriber = AudioTranscriber()
            if transcriber.is_available():
                print(
                    f"Whisper transcription available: model={CONFIG['whisper_model']}"
                )
            else:
                print("Whisper model loading failed")
        except Exception as e:
            print(f"Whisper test failed: {e}")

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
        daemon = SpeechshiftDaemon()
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
