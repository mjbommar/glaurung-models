"""Docker operations manager with proper error handling."""

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Target


class DockerError(Exception):
    """Docker-specific errors."""

    pass


class DockerManager:
    """Manages Docker operations for binary extraction."""

    def __init__(self) -> None:
        """Initialize Docker manager and verify Docker availability."""
        self._verify_docker()
        self._active_containers: List[str] = []

    def _verify_docker(self) -> None:
        """Verify Docker is available and running."""
        if not shutil.which("docker"):
            raise DockerError("Docker command not found. Please install Docker.")

        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Docker daemon not running or accessible: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise DockerError(
                "Docker command timed out. Docker daemon may be unresponsive."
            )

    def pull_image(self, image: str, platform: str) -> bool:
        """
        Pull Docker image for specific platform.

        Args:
            image: Docker image name (e.g., "alpine:3.19")
            platform: Target platform (e.g., "linux/amd64")

        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(
                ["docker", "pull", "--platform", platform, image],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull {image} for {platform}: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print(f"Timeout pulling {image} for {platform}")
            return False

    def create_container(self, image: str, platform: str) -> Optional[str]:
        """
        Create a container without starting it.

        Args:
            image: Docker image name
            platform: Target platform

        Returns:
            Container ID if successful, None otherwise
        """
        container_name = f"binary-extractor-{uuid.uuid4().hex[:8]}"

        try:
            result = subprocess.run(
                [
                    "docker",
                    "create",
                    "--platform",
                    platform,
                    "--name",
                    container_name,
                    image,
                    "sleep",
                    "3600",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )

            result.stdout.strip()
            self._active_containers.append(container_name)
            return container_name

        except subprocess.CalledProcessError as e:
            print(f"Failed to create container for {image}/{platform}: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print(f"Timeout creating container for {image}/{platform}")
            return None

    def copy_filesystem(
        self, container_name: str, temp_dir: Path, target: Target
    ) -> bool:
        """
        Copy entire filesystem from container to local directory.

        Args:
            container_name: Name of the container
            temp_dir: Local directory to copy files to

        Returns:
            True if successful, False otherwise
        """
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Try copying entire filesystem first
        try:
            subprocess.run(
                ["docker", "cp", f"{container_name}:/", str(temp_dir)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for large filesystems
                check=True,
            )
            return True

        except subprocess.CalledProcessError as e:
            # If full copy fails due to permissions, try copying key directories individually
            print(f"Full filesystem copy failed, trying selective copy: {e.stderr}")
            return self._copy_selective_filesystem(container_name, temp_dir, target)

        except subprocess.TimeoutExpired:
            print(f"Timeout copying filesystem from {container_name}")
            return False

    def _copy_selective_filesystem(
        self, container_name: str, temp_dir: Path, target: Target
    ) -> bool:
        """
        Copy key directories individually when full copy fails.

        Args:
            container_name: Name of the container
            temp_dir: Local directory to copy files to
            target: Target configuration with paths to extract

        Returns:
            True if at least some directories copied successfully
        """
        # Extract directory paths from target's path patterns
        # Convert glob patterns like "/usr/bin/**/*" to directory paths like "/usr/bin"
        key_dirs = set()
        for path_pattern in target.paths:
            # Remove glob patterns and get the base directory
            if "**" in path_pattern:
                base_dir = path_pattern.split("**")[0].rstrip("/")
            elif "*" in path_pattern:
                base_dir = path_pattern.split("*")[0].rstrip("/")
            else:
                base_dir = str(Path(path_pattern).parent)

            if base_dir and base_dir != ".":
                key_dirs.add(base_dir)

        key_dirs = list(key_dirs)

        success_count = 0

        for dir_path in key_dirs:
            try:
                # Create unique subdirectory to avoid conflicts
                dir_name = dir_path.replace("/", "_").strip("_")
                dest_path = temp_dir / dir_name

                subprocess.run(
                    ["docker", "cp", f"{container_name}:{dir_path}", str(dest_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )
                success_count += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Silently skip directories that can't be copied due to permissions
                continue

        # Consider successful if we copied at least half the key directories
        return success_count >= 3

    def remove_container(self, container_name: str) -> bool:
        """
        Remove a container.

        Args:
            container_name: Name of the container to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            if container_name in self._active_containers:
                self._active_containers.remove(container_name)
            return True

        except subprocess.CalledProcessError:
            print(f"Failed to remove container {container_name}")
            return False
        except subprocess.TimeoutExpired:
            print(f"Timeout removing container {container_name}")
            return False

    def extract_from_image(
        self, target: Target, platform: str, temp_base_dir: Path
    ) -> Tuple[Optional[Path], Dict[str, any]]:
        """
        Complete extraction process for a single image/platform combination.

        Args:
            target: Target configuration
            platform: Platform to extract
            temp_base_dir: Base directory for temporary files

        Returns:
            Tuple of (temp_filesystem_path, extraction_metadata)
        """
        metadata = {
            "image": target.image,
            "platform": platform,
            "label": target.label,
            "success": False,
            "error": None,
            "container_created": False,
            "filesystem_copied": False,
        }

        # Pull image
        if not self.pull_image(target.image, platform):
            metadata["error"] = "Failed to pull image"
            return None, metadata

        # Create container
        container_name = self.create_container(target.image, platform)
        if not container_name:
            metadata["error"] = "Failed to create container"
            return None, metadata

        metadata["container_created"] = True
        metadata["container_name"] = container_name

        try:
            # Create temp directory for this extraction
            temp_dir = (
                temp_base_dir
                / f"{target.label}-{platform.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
            )

            # Copy filesystem
            if not self.copy_filesystem(container_name, temp_dir, target):
                metadata["error"] = "Failed to copy filesystem"
                return None, metadata

            metadata["filesystem_copied"] = True
            metadata["temp_path"] = str(temp_dir)
            metadata["success"] = True

            return temp_dir, metadata

        finally:
            # Always clean up container
            self.remove_container(container_name)

    def cleanup_all(self) -> None:
        """Clean up all active containers."""
        for container_name in self._active_containers.copy():
            self.remove_container(container_name)

    def __enter__(self) -> "DockerManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup_all()
