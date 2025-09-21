import logging
from pathlib import Path

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path.home() / ".speechshift.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
