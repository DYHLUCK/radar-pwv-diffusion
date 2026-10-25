#!/usr/bin/env python3
"""Standalone entry point. Nothing runs until an explicit command is supplied."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cli import cli
if __name__ == '__main__':
    cli('ddpm')
