"""File operations manager with pathlib and glob support."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .config import Target


class FileManager:
    """Manages file operations for binary extraction."""

    def __init__(self, output_dir: Path, temp_dir: Path) -> None:
        """
        Initialize file manager.

        Args:
            output_dir: Base output directory for extracted binaries
            temp_dir: Base temporary directory for extractions
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def is_executable(self, file_path: Path) -> bool:
        """
        Check if a file is executable.

        Args:
            file_path: Path to check

        Returns:
            True if file is executable
        """
        try:
            return file_path.is_file() and os.access(file_path, os.X_OK)
        except (OSError, PermissionError):
            return False

    def is_binary(self, file_path: Path) -> bool:
        """
        Check if a file appears to be a binary (not text).

        Args:
            file_path: Path to check

        Returns:
            True if file appears to be binary
        """
        if not file_path.is_file():
            return False

        try:
            # Read first 1024 bytes to check for null bytes
            with file_path.open("rb") as f:
                chunk = f.read(1024)
                return b"\x00" in chunk
        except (OSError, PermissionError, UnicodeDecodeError):
            return True  # If we can't read it, assume it's binary

    def matches_patterns(
        self, file_path: Path, include_patterns: List[str], exclude_patterns: List[str]
    ) -> bool:
        """
        Check if a file matches include/exclude patterns.

        Args:
            file_path: Path to check
            include_patterns: Patterns that must match
            exclude_patterns: Patterns that must not match

        Returns:
            True if file matches criteria
        """

        # Check exclude patterns first
        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return False

        # Check include patterns
        for pattern in include_patterns:
            if file_path.match(pattern):
                return True

        return False

    def find_executables(
        self, temp_fs_dir: Path, target: Target
    ) -> Iterator[Tuple[Path, str]]:
        """
        Find executable files in extracted filesystem using glob patterns.

        Args:
            temp_fs_dir: Directory containing extracted filesystem
            target: Target configuration with path patterns

        Yields:
            Tuple of (file_path, original_container_path)
        """
        for path_pattern in target.paths:
            # Remove leading slash and convert to glob pattern
            clean_pattern = path_pattern.lstrip("/")
            temp_fs_dir / clean_pattern

            # Use glob to find matching files
            for file_path in temp_fs_dir.glob(clean_pattern):
                if (
                    file_path.is_file()
                    and self.is_executable(file_path)
                    and self.is_binary(file_path)
                ):
                    # Check include/exclude patterns
                    if self.matches_patterns(
                        file_path, target.include_patterns, target.exclude_patterns
                    ):
                        # Calculate original container path
                        relative_path = file_path.relative_to(temp_fs_dir)
                        container_path = "/" + str(relative_path)
                        yield file_path, container_path

    def organize_binaries(
        self, temp_fs_dir: Path, target: Target, platform: str
    ) -> Dict[str, any]:
        """
        Organize extracted binaries into structured output directory.

        Args:
            temp_fs_dir: Directory containing extracted filesystem
            target: Target configuration
            platform: Platform string (e.g., "linux/amd64")

        Returns:
            Dictionary with organization results
        """
        # Create platform-specific output directory
        platform_dir = (
            self.output_dir / target.label / self._platform_to_dirname(platform)
        )
        platform_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "target": target.label,
            "platform": platform,
            "output_directory": str(platform_dir),
            "total_files_found": 0,
            "executables_copied": 0,
            "paths_processed": {},
            "errors": [],
        }

        # Find and copy executables
        for file_path, container_path in self.find_executables(temp_fs_dir, target):
            results["total_files_found"] += 1

            try:
                # Create subdirectory based on original path
                container_dir = Path(container_path).parent
                dest_subdir = platform_dir / self._sanitize_path(container_dir)
                dest_subdir.mkdir(parents=True, exist_ok=True)

                # Copy file
                dest_file = dest_subdir / file_path.name
                shutil.copy2(file_path, dest_file)

                # Track results
                results["executables_copied"] += 1
                container_dir_str = str(container_dir)
                if container_dir_str not in results["paths_processed"]:
                    results["paths_processed"][container_dir_str] = 0
                results["paths_processed"][container_dir_str] += 1

            except (OSError, PermissionError, shutil.Error) as e:
                error_msg = f"Failed to copy {container_path}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"Warning: {error_msg}")

        # Save metadata
        self._save_extraction_metadata(platform_dir, target, platform, results)

        return results

    def _platform_to_dirname(self, platform: str) -> str:
        """Convert platform string to directory name."""
        return platform.replace("/", "-").replace(":", "-")

    def _sanitize_path(self, path: Path) -> str:
        """Sanitize path for use as directory name."""
        # Convert absolute path to relative and replace problematic chars
        path_str = str(path).lstrip("/")
        return path_str.replace("/", "-").replace(":", "-")

    def _save_extraction_metadata(
        self, output_dir: Path, target: Target, platform: str, results: Dict
    ) -> None:
        """Save extraction metadata to JSON file."""
        metadata = {
            "extraction_info": {
                "image": target.image,
                "platform": platform,
                "label": target.label,
                "description": target.description,
                "extracted_at": datetime.utcnow().isoformat() + "Z",
            },
            "configuration": {
                "paths": target.paths,
                "include_patterns": target.include_patterns,
                "exclude_patterns": target.exclude_patterns,
            },
            "results": results,
        }

        metadata_file = output_dir / "extraction_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    def cleanup_temp(self, temp_dir: Path) -> bool:
        """
        Clean up temporary directory.

        Args:
            temp_dir: Directory to remove

        Returns:
            True if successful
        """
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return True
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")
            return False

    def get_extraction_summary(self) -> Dict[str, any]:
        """
        Generate summary of all extractions.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_distributions": 0,
            "total_architectures": 0,
            "total_executables": 0,
            "total_size_bytes": 0,
            "distributions": {},
        }

        # Walk through output directory
        for dist_dir in self.output_dir.iterdir():
            if dist_dir.is_dir():
                summary["total_distributions"] += 1
                dist_info = {
                    "architectures": {},
                    "total_executables": 0,
                    "total_size_bytes": 0,
                }

                for arch_dir in dist_dir.iterdir():
                    if arch_dir.is_dir():
                        # Count executables and size
                        exec_count = 0
                        total_size = 0

                        for file_path in arch_dir.rglob("*"):
                            if file_path.is_file() and self.is_executable(file_path):
                                exec_count += 1
                                try:
                                    total_size += file_path.stat().st_size
                                except OSError:
                                    pass

                        dist_info["architectures"][arch_dir.name] = {
                            "executables": exec_count,
                            "size_bytes": total_size,
                        }

                        dist_info["total_executables"] += exec_count
                        dist_info["total_size_bytes"] += total_size

                summary["distributions"][dist_dir.name] = dist_info
                summary["total_executables"] += dist_info["total_executables"]
                summary["total_size_bytes"] += dist_info["total_size_bytes"]

        summary["total_architectures"] = len(
            {
                arch
                for dist_info in summary["distributions"].values()
                for arch in dist_info["architectures"].keys()
            }
        )

        return summary

    def save_summary_report(self, summary: Dict[str, any]) -> Path:
        """Save summary report to JSON file."""
        report_path = self.output_dir / "extraction_summary.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return report_path
