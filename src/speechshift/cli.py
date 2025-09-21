#!/usr/bin/env python3
"""
CLI entry point for SpeechShift
"""

import sys
from .main import main as speechshift_main


def main():
    """Main CLI entry point"""
    sys.exit(speechshift_main())


if __name__ == "__main__":
    main()
