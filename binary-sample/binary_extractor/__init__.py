"""
Binary Extractor - Modern Docker-based binary collection system.

A clean, efficient system for extracting system binaries from Docker images
across multiple architectures using filesystem copy approach.
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .config import ExtractionConfig, Target
from .extractor import BinaryExtractor
from .docker_manager import DockerManager
from .file_manager import FileManager

__all__ = [
    "ExtractionConfig",
    "Target",
    "BinaryExtractor",
    "DockerManager",
    "FileManager",
]
