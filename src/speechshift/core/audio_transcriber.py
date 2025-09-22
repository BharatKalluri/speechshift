import os
import re
from pathlib import Path
from typing import Optional

import assemblyai as aai
import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from abc import ABC, abstractmethod

from speechshift.core.config import CONFIG
from speechshift.core.logger import logger


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_file: Path) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError


class WhisperTranscriber(Transcriber):
    def __init__(self):
        self.model = None
        self.model_size = CONFIG["whisper_model"]
        self.device = CONFIG["whisper_device"]
        self.compute_type = CONFIG["whisper_compute_type"]
        self.language = CONFIG["whisper_language"]

        self.beam_size = 5
        self.temperature = 0.0
        self.repetition_penalty = 1.1
        self.no_repeat_ngram_size = 2
        self.condition_on_previous_text = False
        self.word_timestamps = False
        self.vad_filter = True
        self.vad_threshold = 0.5
        self.silence_duration_ms = 300
        self.trim_top_db = 20

    def _load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            self.model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def is_available(self) -> bool:
        return self._load_model()

    def transcribe(self, audio_file: Path) -> Optional[str]:
        if not self._load_model():
            return None

        audio_data, sample_rate = self._load_audio_data(audio_file)
        if audio_data is None:
            return None

        normalized_audio = self._normalize_audio(audio_data)
        preprocessed_file = audio_file.parent / f"preprocessed_{audio_file.name}"
        self._save_preprocessed_audio(normalized_audio, sample_rate, preprocessed_file)

        try:
            return self._transcribe_with_whisper(preprocessed_file)
        finally:
            try:
                if preprocessed_file.exists():
                    preprocessed_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up preprocessed file: {e}")

    def _transcribe_with_whisper(self, preprocessed_file: Path) -> Optional[str]:
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

        transcribed_text = "".join(segment.text for segment in segments).strip()
        logger.info(f"Before postprocessing: {transcribed_text}")

        if transcribed_text:
            processed_text = self._post_process_text(transcribed_text)
            logger.info(f"Transcription completed: '{processed_text}'")
            return processed_text
        else:
            logger.warning("Transcription resulted in empty text")
            return None

    @staticmethod
    def _load_audio_data(audio_file: Path):
        audio_data, sample_rate = librosa.load(str(audio_file), sr=None, mono=True)
        return audio_data, sample_rate

    @staticmethod
    def _normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        normalized = librosa.util.normalize(audio_data, norm=np.inf, axis=None)
        return normalized * 0.8

    @staticmethod
    def _save_preprocessed_audio(
        audio_data: np.ndarray, sample_rate: int, output_path: Path
    ):
        sf.write(str(output_path), audio_data, sample_rate)

    @staticmethod
    def _post_process_text(text: str) -> str:
        if not text or not text.strip():
            return ""

        original_text = text
        text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)

        for phrase_len in range(8, 1, -1):
            pattern = r"\b((?:\w+\s+){" + str(phrase_len - 1) + r"}\w+)(?:\s+\1){1,}"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
        sentences = text.split(".")
        seen_sentences = set()
        unique_sentences = []
        for i, sentence in enumerate(sentences):
            clean_sentence = sentence.strip().lower()
            if (
                clean_sentence
                and clean_sentence not in seen_sentences
                and len(clean_sentence) > 10
            ):
                seen_sentences.add(clean_sentence)
                unique_sentences.append(sentences[i])
            elif clean_sentence and len(clean_sentence) <= 10:
                unique_sentences.append(sentences[i])
        text = ". ".join(s.strip() for s in unique_sentences if s.strip())
        filler_words = [
            r"\b(?:um+|uh+|ah+|er+|eh+)\b",
            r"\b(?:like|you know|sort of|kind of)\b(?=\s)",
            r"\b(?:actually|basically|literally)\b(?=\s)",
        ]
        for pattern in filler_words:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([,.!?;:])\s*([,.!?;:])", r"\1", text)
        text = re.sub(r"([.!?])\s*([.!?])+", r"\1", text)
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([,;:])([^\s])", r"\1 \2", text)
        text = re.sub(r'\s*"\s*', '"', text)
        text = re.sub(r"\s*\'\s*", "'", text)
        text = re.sub(
            r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )
        text = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), text)
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
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        text = text.strip()
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) * 0.3:
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
        text = re.sub(r"\[\s*\]|\(\s*\)", "", text)
        if text and not text.endswith((".", "!", "?")):
            text += "."
        text = " ".join(text.split())
        if len(text) < len(original_text) * 0.2 and len(original_text) > 50:
            return original_text.strip()
        return text


class AssemblyAITranscriber(Transcriber):
    def __init__(self):
        self.assembly_ai_key = os.environ.get("ASSEMBLYAI_API_KEY")
        self.language = CONFIG["whisper_language"]
        if self.assembly_ai_key:
            aai.settings.api_key = self.assembly_ai_key

    def is_available(self) -> bool:
        return bool(self.assembly_ai_key)

    def transcribe(self, audio_file: Path) -> Optional[str]:
        if not self.is_available():
            logger.error("Assembly AI API key not found.")
            return None

        try:
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.universal,
                language_detection=True if not self.language else False,
                language_code=self.language if self.language else None,
                punctuate=True,
                format_text=True,
            )
            transcriber = aai.Transcriber(config=config)
            logger.info(f"Starting Assembly AI transcription of: {audio_file}")
            transcript = transcriber.transcribe(str(audio_file))

            if transcript.status == aai.TranscriptStatus.error:
                logger.error(f"Assembly AI transcription failed: {transcript.error}")
                return None

            if not transcript.text or not transcript.text.strip():
                logger.warning("Assembly AI transcription resulted in empty text")
                return None

            processed_text = self._post_process_text(transcript.text)
            logger.info(f"Assembly AI transcription completed: '{processed_text}'")
            return processed_text
        except Exception as e:
            logger.error(f"Assembly AI transcription failed: {e}")
            return None

    @staticmethod
    def _post_process_text(text: str) -> str:
        # This can be a shared function if the post-processing is identical.
        # For now, duplicating it for clarity.
        return WhisperTranscriber._post_process_text(text)


class AudioTranscriber:
    def __init__(self):
        self.language = CONFIG["whisper_language"]
        self.timeout = CONFIG["transcription_timeout"]
        self.engines = {
            "whisper": WhisperTranscriber(),
            "assemblyai": AssemblyAITranscriber(),
        }

    def transcribe_audio(
        self, audio_file: Path, engine: str = "whisper"
    ) -> Optional[str]:
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            return None

        engine_lower = engine.lower()
        if engine_lower not in self.engines:
            logger.error(f"Unsupported transcription engine: {engine}")
            return None

        transcriber = self.engines[engine_lower]
        if not transcriber.is_available():
            logger.error(f"{engine} is not available for transcription.")
            return None

        return transcriber.transcribe(audio_file)

    def is_available(self, engine: str) -> bool:
        engine_lower = engine.lower()
        if engine_lower in self.engines:
            return self.engines[engine_lower].is_available()
        return False

    def available_engines(self) -> list[str]:
        return [
            engine
            for engine, transcriber in self.engines.items()
            if transcriber.is_available()
        ]
