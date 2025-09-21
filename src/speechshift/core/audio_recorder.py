import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from speechshift.core.notification_manager import NotificationManager
from speechshift.core.whisper_transcriber import WhisperTranscriber
from speechshift.core.config import CONFIG
from speechshift.core.logger import logger
from speechshift.core.wayland_text_input import WaylandTextInput


class AudioRecorder:
    """Handle audio recording with PipeWire via sounddevice"""

    def __init__(self):
        self.sample_rate = CONFIG["sample_rate"]
        self.channels = CONFIG["channels"]
        self.dtype = CONFIG["dtype"]

    def get_default_device(self) -> Optional[int]:
        """Get default input device"""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]  # Input device
            return default_input
        except Exception as e:
            logger.error(f"Failed to get default device: {e}")
            return None

    def start_recording(self) -> bool:
        """Start audio recording"""
        # Check if already recording
        if recording_state.is_recording:
            logger.warning("Recording already in progress")
            return False

        try:
            # Create temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = CONFIG["temp_dir"] / f"recording_{timestamp}.wav"

            # Initialize recording state
            recording_state.is_recording = True
            recording_state.audio_data = []
            recording_state.current_file = temp_file
            recording_state.start_time = time.time()

            # Start recording thread
            recording_state.recording_thread = threading.Thread(
                target=self._recording_worker, args=(temp_file,)
            )
            recording_state.recording_thread.start()

            NotificationManager.recording_started()
            logger.info(f"Recording started: {temp_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            NotificationManager.recording_error(str(e))
            recording_state.is_recording = False
            return False

    def stop_recording(self) -> Optional[str]:
        """Stop audio recording and transcribe"""
        # Simple state check - no recording means nothing to stop
        if not recording_state.is_recording:
            logger.warning("No recording in progress")
            return None

        # Check if already transcribing
        if recording_state.is_transcribing:
            logger.info("Transcription already in progress")
            return None

        # Mark as transcribing to prevent duplicate calls
        recording_state.is_transcribing = True

        try:
            # Stop the recording
            self._stop_recording_process()

            # Get the recorded file
            temp_file = recording_state.current_file
            if not temp_file or not temp_file.exists():
                logger.warning("No recording file found")
                return None

            # Transcribe and handle result
            result = self._transcribe_and_insert(temp_file)

            # Clean up
            self._cleanup_temp_file(temp_file)

            return result

        except Exception as e:
            logger.error(f"Stop recording failed: {e}")
            NotificationManager.recording_error(str(e))
            return None
        finally:
            # Always reset state
            self._reset_recording_state()

    def _stop_recording_process(self):
        """Stop the actual recording process"""
        recording_state.is_recording = False

        # Wait for recording thread to finish
        if recording_state.recording_thread:
            recording_state.recording_thread.join(timeout=5.0)
            if recording_state.recording_thread.is_alive():
                logger.warning("Recording thread did not stop cleanly")

        NotificationManager.recording_stopped(
            recording_state.current_file.name
            if recording_state.current_file
            else "unknown"
        )
        logger.info("Recording stopped")

    def _transcribe_and_insert(self, temp_file: Path) -> Optional[str]:
        """Transcribe audio file and insert text"""
        NotificationManager.transcription_started()

        try:
            transcriber = WhisperTranscriber()
            transcribed_text = transcriber.transcribe_audio(temp_file)

            if transcribed_text:
                WaylandTextInput.insert_text(transcribed_text)
                NotificationManager.transcription_completed(transcribed_text)
                logger.info(f"Transcription successful: {transcribed_text}")
                return transcribed_text
            else:
                WaylandTextInput.insert_text("(transcription failed)")
                NotificationManager.transcription_error("Empty transcription")
                return None

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            WaylandTextInput.insert_text("(transcription error)")
            NotificationManager.transcription_error(str(e))
            return None

    def _cleanup_temp_file(self, temp_file: Path):
        """Clean up temporary recording file"""
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.info(f"Temporary file cleaned up: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")

    def _reset_recording_state(self):
        """Reset all recording state variables"""
        recording_state.current_file = None
        recording_state.audio_data = []
        recording_state.recording_thread = None
        recording_state.is_transcribing = False

    def _recording_worker(self, output_file: Path):
        """Worker thread for audio recording"""
        try:
            with wave.open(str(output_file), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)

                def audio_callback(indata, frames, time, status):
                    if status:
                        logger.warning(f"Audio callback status: {status}")

                    if recording_state.is_recording:
                        # Convert float32 to int16
                        audio_int16 = (indata * 32767).astype(np.int16)
                        wf.writeframes(audio_int16.tobytes())

                # Start audio stream
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                ):
                    while recording_state.is_recording:
                        time.sleep(0.1)

        except Exception as e:
            logger.error(f"Recording worker error: {e}")
            recording_state.is_recording = False


class RecordingState:
    def __init__(self):
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = []
        self.current_file = None
        self.start_time = None
        self.is_transcribing = False


recording_state = RecordingState()
