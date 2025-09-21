import subprocess

from speechshift.core.logger import logger


class NotificationManager:
    """Handle mako notifications"""

    @staticmethod
    def send_notification(
        title: str, message: str, urgency: str = "normal", timeout: int = 3000
    ):
        """Send notification via notify-send"""
        try:
            subprocess.run(
                [
                    "notify-send",
                    "--urgency",
                    urgency,
                    "--expire-time",
                    str(timeout),
                    "--app-name",
                    "SpeechShift",
                    title,
                    message,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to send notification: {e}")

    @classmethod
    def recording_started(cls):
        """Show recording started notification"""
        cls.send_notification("🎤 SpeechShift", "Recording started...", "normal")

    @classmethod
    def recording_stopped(cls, filename: str):
        """Show recording stopped notification"""
        cls.send_notification(
            "✅ SpeechShift", f"Recording saved: {filename}", "normal"
        )

    @classmethod
    def recording_error(cls, error: str):
        """Show recording error notification"""
        cls.send_notification(
            "❌ SpeechShift", f"Recording failed: {error}", "critical"
        )

    @classmethod
    def transcription_started(cls):
        """Show transcription started notification"""
        cls.send_notification("🔄 SpeechShift", "Transcribing audio...", "normal")

    @classmethod
    def transcription_completed(cls, text: str):
        """Show transcription completed notification"""
        preview = text[:50] + "..." if len(text) > 50 else text
        cls.send_notification("✅ SpeechShift", f"Transcribed: {preview}", "normal")

    @classmethod
    def transcription_error(cls, error: str):
        """Show transcription error notification"""
        cls.send_notification(
            "❌ SpeechShift", f"Transcription failed: {error}", "critical"
        )
