import re
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from speechshift.core.config import CONFIG
from speechshift.core.logger import logger


class WhisperTranscriber:
    """
    Handle audio transcription using faster-whisper
    """

    def __init__(self):
        self.model = None
        self.model_size = CONFIG["whisper_model"]
        self.device = CONFIG["whisper_device"]
        self.compute_type = CONFIG["whisper_compute_type"]
        self.language = CONFIG["whisper_language"]
        self.timeout = CONFIG["transcription_timeout"]

        self.beam_size = 5  # Beam search size (higher = more accurate but slower)
        self.temperature = 0.0  # Temperature for sampling (0.0 = deterministic)
        self.repetition_penalty = 1.1  # Penalty for repetitions
        self.no_repeat_ngram_size = 2  # Prevent n-gram repetition
        self.condition_on_previous_text = False  # Better for short clips
        self.word_timestamps = False  # Disable word-level timestamps
        self.vad_filter = True  # Use built-in VAD if available
        self.vad_threshold = 0.5  # VAD threshold (0.0-1.0)
        self.silence_duration_ms = 300  # Minimum silence duration in ms

        # Librosa audio preprocessing parameters
        self.trim_top_db = 20  # dB threshold for silence trimming

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

    @staticmethod
    def _load_audio_data(audio_file: Path):
        """Load audio data using librosa (supports multiple formats)"""
        audio_data, sample_rate = librosa.load(
            str(audio_file),
            sr=None,  # Keep original sample rate
            mono=True,  # Convert to mono
        )
        return audio_data, sample_rate

    def _trim_silence(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Trim silence from beginning and end using librosa"""
        trimmed_audio, _ = librosa.effects.trim(
            audio_data, top_db=self.trim_top_db, frame_length=2048, hop_length=512
        )

        if len(trimmed_audio) > 0:
            logger.info(
                f"Trimmed audio from {len(audio_data)} to {len(trimmed_audio)} samples"
            )
            return trimmed_audio
        else:
            logger.warning("Trimming resulted in empty audio, using original")
            return audio_data

    @staticmethod
    def _normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio using librosa"""
        normalized = librosa.util.normalize(audio_data, norm=np.inf, axis=None)
        return normalized * 0.8

    @staticmethod
    def _save_preprocessed_audio(
        audio_data: np.ndarray, sample_rate: int, output_path: Path
    ):
        """Save preprocessed audio using soundfile"""
        sf.write(str(output_path), audio_data, sample_rate)

    @staticmethod
    def _post_process_text(text: str) -> str:
        """Apply post-processing corrections to transcribed text"""
        if not text or not text.strip():
            return ""

        original_text = text

        # 1. HALLUCINATION DETECTION - Remove common Whisper artifacts TODO

        # 2. REPETITION REMOVAL - Major Whisper issue

        # Remove word repetitions (3+ consecutive identical words)
        text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)

        # Remove phrase repetitions (2+ consecutive identical phrases of 2-8 words)
        for phrase_len in range(8, 1, -1):  # Start with longer phrases
            pattern = r"\b((?:\w+\s+){" + str(phrase_len - 1) + r"}\w+)(?:\s+\1){1,}"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        # Remove sentence repetitions
        sentences = text.split(".")
        seen_sentences = set()
        unique_sentences = []

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if sentence and sentence not in seen_sentences and len(sentence) > 10:
                seen_sentences.add(sentence)
                unique_sentences.append(sentences[len(unique_sentences)])
            elif sentence and len(sentence) <= 10:  # Keep short sentences
                unique_sentences.append(sentences[len(unique_sentences)])

        text = ". ".join(s.strip() for s in unique_sentences if s.strip())

        # 3. FILLER WORD REMOVAL
        filler_words = [
            r"\b(?:um+|uh+|ah+|er+|eh+)\b",
            r"\b(?:like|you know|sort of|kind of)\b(?=\s)",
            r"\b(?:actually|basically|literally)\b(?=\s)",
        ]

        for pattern in filler_words:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 4. PUNCTUATION FIXES

        # Fix spacing around punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(
            r"([,.!?;:])\s*([,.!?;:])", r"\1", text
        )  # Remove double punctuation
        text = re.sub(
            r"([.!?])\s*([.!?])+", r"\1", text
        )  # Remove repeated end punctuation

        # Add spaces after punctuation if missing
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([,;:])([^\s])", r"\1 \2", text)

        # Fix quotation marks
        text = re.sub(r'\s*"\s*', '"', text)
        text = re.sub(r"\s*\'\s*", "'", text)

        # 5. CAPITALIZATION FIXES

        # Capitalize after sentence endings
        text = re.sub(
            r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )

        # Capitalize first word
        text = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), text)

        # Fix common proper nouns (add more as needed)
        proper_nouns = {
            "i": "I",
            "ai": "AI",
            "api": "API",
            "ui": "UI",
            "ux": "UX",
            "cpu": "CPU",
            "gpu": "GPU",
            "ram": "RAM",
            "ssd": "SSD",
            "usa": "USA",
            "uk": "UK",
            "eu": "EU",
        }

        for wrong, correct in proper_nouns.items():
            text = re.sub(
                r"\b" + re.escape(wrong) + r"\b", correct, text, flags=re.IGNORECASE
            )

        # 6. WHITESPACE CLEANUP

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # 7. QUALITY CHECKS - Detect likely hallucinations

        # Check for excessive repetition (might indicate hallucination)
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only count significant words
                    word_counts[word] = word_counts.get(word, 0) + 1

            # If any word appears more than 30% of the time, likely hallucination
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) * 0.3:
                # Try to salvage by keeping only first occurrence of repeated content
                seen_words = set()
                filtered_words = []
                repetition_threshold = 3

                for word in words:
                    if (
                        word not in seen_words
                        or word_counts.get(word, 0) <= repetition_threshold
                    ):
                        filtered_words.append(word)
                        seen_words.add(word)

                text = " ".join(filtered_words)

        # 8. FINAL CLEANUP

        # Remove empty parentheses/brackets
        text = re.sub(r"\[\s*\]|\(\s*\)", "", text)

        # Ensure sentence endings
        if text and not text.endswith((".", "!", "?")):
            text += "."

        # Final whitespace cleanup
        text = " ".join(text.split())

        # If post-processing removed >80% of content, return original (likely over-processed)
        if len(text) < len(original_text) * 0.2 and len(original_text) > 50:
            return original_text.strip()

        return text

    def transcribe_audio(self, audio_file: Path) -> Optional[str]:
        """Transcribe audio file to text with preprocessing and optimization"""
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            return None

        # Load model if not already loaded
        if not self._load_model():
            return None

        logger.info(f"Starting transcription of: {audio_file}")

        # Load and preprocess audio
        audio_data, sample_rate = self._load_audio_data(audio_file)
        if audio_data is None:
            return None

        # Apply preprocessing
        logger.info("Applying audio preprocessing...")
        trimmed_audio = self._trim_silence(audio_data, sample_rate)
        normalized_audio = self._normalize_audio(trimmed_audio)

        # Save preprocessed audio to temporary file
        preprocessed_file = audio_file.parent / f"preprocessed_{audio_file.name}"
        self._save_preprocessed_audio(normalized_audio, sample_rate, preprocessed_file)

        try:
            # Perform transcription
            segments, info = self.model.transcribe(
                str(preprocessed_file),
                language=self.language,
                beam_size=self.beam_size,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                condition_on_previous_text=self.condition_on_previous_text,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
                vad_parameters=(
                    dict(
                        threshold=self.vad_threshold,
                        min_silence_duration_ms=self.silence_duration_ms,
                    )
                    if self.vad_filter
                    else None
                ),
            )

            logger.info(
                f"Detected language: {info.language} (confidence: {info.language_probability:.2f})"
            )

            # Collect all text segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text

            # Clean up and post-process the text
            transcribed_text = transcribed_text.strip()
            logger.info(f"Before postprocessing: {transcribed_text}")
            if transcribed_text:
                processed_text = self._post_process_text(transcribed_text)
                logger.info(f"Transcription completed: '{processed_text}...'")
                return processed_text
            else:
                logger.warning("Transcription resulted in empty text")
                return None
        finally:
            # Clean up preprocessed file
            try:
                if preprocessed_file.exists():
                    preprocessed_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up preprocessed file: {e}")

    def is_available(self) -> bool:
        """Check if transcription is available"""
        return self._load_model()
