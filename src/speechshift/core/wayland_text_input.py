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
    def get_clipboard() -> str:
        """Get clipboard content using wl-paste"""
        try:
            result = subprocess.run(
                ["wl-paste"], check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get clipboard: {e}")
            return ""

    @classmethod
    def paste_text(cls, text: str) -> bool:
        """Paste text by setting clipboard and simulating Ctrl+V"""
        original_clipboard_content = cls.get_clipboard()
        try:
            # Set clipboard
            if not cls.set_clipboard(text):
                return False

            # Small delay to ensure clipboard is set
            time.sleep(0.5)

            # Simulate Ctrl+V
            subprocess.run(
                ["hyprctl", "dispatch", "sendshortcut", "CTRL,V,"], check=True
            )
            # Small delay to ensure that paste went through
            time.sleep(0.5)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to paste text: {e}")
            return False
        finally:
            # Restore original clipboard content
            cls.set_clipboard(original_clipboard_content)

    @classmethod
    def insert_text(cls, text: str) -> bool:
        """Insert text using the best available method"""
        # Try direct typing first
        # if cls.type_text(text):
        #     return True

        # Fallback to clipboard paste
        # logger.info("Direct typing failed, trying clipboard paste")
        return cls.paste_text(text)
