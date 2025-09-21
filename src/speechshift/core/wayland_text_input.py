import subprocess
import time

from speechshift.core.logger import logger


class WaylandTextInput:
    """Handle Wayland text input via wl-clipboard and wtype"""

    @staticmethod
    def set_clipboard(text: str) -> bool:
        """Set clipboard content using wl-copy"""
        try:
            subprocess.run(["wl-copy", text], check=True, input=text.encode())
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set clipboard: {e}")
            return False

    @staticmethod
    def type_text(text: str) -> bool:
        """Type text using wtype"""
        try:
            subprocess.run(["wtype", text], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to type text: {e}")
            return False

    @classmethod
    def paste_text(cls, text: str) -> bool:
        """Paste text by setting clipboard and simulating Ctrl+V"""
        try:
            # Set clipboard
            if not cls.set_clipboard(text):
                return False

            # Small delay to ensure clipboard is set
            time.sleep(0.1)

            # Simulate Ctrl+V
            subprocess.run(["wtype", "-M", "ctrl", "v"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to paste text: {e}")
            return False

    @classmethod
    def insert_text(cls, text: str) -> bool:
        """Insert text using the best available method"""
        # Try direct typing first
        if cls.type_text(text):
            return True

        # Fallback to clipboard paste
        logger.info("Direct typing failed, trying clipboard paste")
        return cls.paste_text(text)
