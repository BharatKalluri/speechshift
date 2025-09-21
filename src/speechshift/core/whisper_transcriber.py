from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from speechshift.core.config import CONFIG
from speechshift.core.logger import logger


class WhisperTranscriber:
    """Handle audio transcription using faster-whisper"""

    def __init__(self):
        self.model = None
        self.model_size = CONFIG["whisper_model"]
        self.device = CONFIG["whisper_device"]
        self.compute_type = CONFIG["whisper_compute_type"]
        self.language = CONFIG["whisper_language"]
        self.timeout = CONFIG["transcription_timeout"]

    def _load_model(self) -> bool:
        """Load the Whisper model (lazy loading)"""
        if self.model is not None:
            return True

        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe_audio(self, audio_file: Path) -> Optional[str]:
        """Transcribe audio file to text"""
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            return None

        try:
            # Load model if not already loaded
            if not self._load_model():
                return None

            logger.info(f"Starting transcription of: {audio_file}")

            # Perform transcription
            segments, info = self.model.transcribe(
                str(audio_file),
                language=self.language,
                beam_size=5,
                word_timestamps=False,
            )

            # Collect all text segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text

            # Clean up the text
            transcribed_text = transcribed_text.strip()

            if transcribed_text:
                logger.info(f"Transcription completed: '{transcribed_text[:50]}...'")
                return transcribed_text
            else:
                logger.warning("Transcription resulted in empty text")
                return None

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def is_available(self) -> bool:
        """Check if transcription is available"""
        return self._load_model()
