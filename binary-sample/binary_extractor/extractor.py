"""Main binary extractor orchestration class."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import ExtractionConfig, Target
from .docker_manager import DockerManager
from .file_manager import FileManager


class BinaryExtractor:
    """Main orchestrator for binary extraction process."""

    def __init__(self, config: ExtractionConfig) -> None:
        """
        Initialize binary extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config
        self.settings = config.extraction_config

        # Initialize managers
        self.file_manager = FileManager(
            output_dir=self.settings.output_directory,
            temp_dir=self.settings.temp_directory,
        )

        # Setup logging
        self._setup_logging()

        # Track extraction results
        self.extraction_results: List[Dict] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.settings.output_directory / "extraction.log"),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def extract_single_target(self, target: Target, platform: str) -> Dict:
        """
        Extract binaries from a single target/platform combination.

        Args:
            target: Target configuration
            platform: Platform to extract

        Returns:
            Dictionary with extraction results
        """
        self.logger.info(f"Starting extraction: {target.image} ({platform})")

        extraction_start = time.time()
        result = {
            "target": target.label,
            "image": target.image,
            "platform": platform,
            "start_time": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "executables_found": 0,
            "errors": [],
        }

        try:
            with DockerManager() as docker:
                # Extract filesystem from container
                temp_fs_dir, docker_metadata = docker.extract_from_image(
                    target, platform, self.settings.temp_directory
                )

                if not docker_metadata["success"]:
                    result["errors"].append(
                        docker_metadata.get("error", "Unknown Docker error")
                    )
                    result["docker_metadata"] = docker_metadata
                    return result

                # Organize binaries
                self.logger.info(f"Organizing binaries for {target.image} ({platform})")
                organization_results = self.file_manager.organize_binaries(
                    temp_fs_dir, target, platform
                )

                result.update(
                    {
                        "success": True,
                        "executables_found": organization_results["executables_copied"],
                        "total_files_processed": organization_results[
                            "total_files_found"
                        ],
                        "output_directory": organization_results["output_directory"],
                        "paths_processed": organization_results["paths_processed"],
                        "errors": organization_results["errors"],
                    }
                )

                # Cleanup temp directory if configured
                if self.settings.cleanup_temp and temp_fs_dir:
                    self.file_manager.cleanup_temp(temp_fs_dir)

                self.logger.info(
                    f"Completed extraction: {target.image} ({platform}) - "
                    f"{result['executables_found']} executables found"
                )

        except Exception as e:
            error_msg = (
                f"Unexpected error extracting {target.image} ({platform}): {str(e)}"
            )
            self.logger.error(error_msg)
            result["errors"].append(error_msg)

        finally:
            extraction_time = time.time() - extraction_start
            result["extraction_time_seconds"] = round(extraction_time, 2)
            result["end_time"] = datetime.utcnow().isoformat() + "Z"

        return result

    def extract_all_targets(self, max_workers: Optional[int] = None) -> None:
        """
        Extract binaries from all configured targets using concurrent execution.

        Args:
            max_workers: Maximum number of concurrent extractions (defaults to config)
        """
        if max_workers is None:
            max_workers = self.settings.max_concurrent_extractions

        self.start_time = time.time()
        self.logger.info(
            f"Starting extraction of {len(self.config.targets)} targets with {max_workers} workers"
        )

        # Create list of all target/platform combinations
        extraction_tasks = []
        for target in self.config.targets:
            for platform in target.platforms:
                extraction_tasks.append((target, platform))

        total_tasks = len(extraction_tasks)
        self.logger.info(f"Total extraction tasks: {total_tasks}")

        # Execute extractions concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.extract_single_target, target, platform): (
                    target,
                    platform,
                )
                for target, platform in extraction_tasks
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                target, platform = future_to_task[future]

                try:
                    result = future.result()
                    self.extraction_results.append(result)

                    status = "✅" if result["success"] else "❌"
                    self.logger.info(
                        f"{status} ({completed}/{total_tasks}) {target.image} ({platform}) - "
                        f"{result.get('executables_found', 0)} executables"
                    )

                except Exception as e:
                    error_result = {
                        "target": target.label,
                        "image": target.image,
                        "platform": platform,
                        "success": False,
                        "errors": [f"Task execution failed: {str(e)}"],
                    }
                    self.extraction_results.append(error_result)
                    self.logger.error(
                        f"❌ ({completed}/{total_tasks}) {target.image} ({platform}) - FAILED: {e}"
                    )

        self.end_time = time.time()
        self.logger.info(
            f"Completed all extractions in {self.end_time - self.start_time:.1f} seconds"
        )

    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report.

        Returns:
            Dictionary with complete extraction summary
        """
        if not self.extraction_results:
            return {"error": "No extraction results available"}

        # Calculate overall statistics
        successful_extractions = [
            r for r in self.extraction_results if r.get("success", False)
        ]
        failed_extractions = [
            r for r in self.extraction_results if not r.get("success", False)
        ]

        total_executables = sum(
            r.get("executables_found", 0) for r in successful_extractions
        )
        total_time = (
            (self.end_time - self.start_time)
            if (self.end_time and self.start_time)
            else 0
        )

        # Get file system summary
        file_summary = self.file_manager.get_extraction_summary()

        summary = {
            "extraction_summary": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_targets": len(self.config.targets),
                "total_tasks": len(self.extraction_results),
                "successful_extractions": len(successful_extractions),
                "failed_extractions": len(failed_extractions),
                "total_executables_found": total_executables,
                "total_extraction_time_seconds": round(total_time, 2),
                "average_extraction_time": round(
                    total_time / len(self.extraction_results), 2
                )
                if self.extraction_results
                else 0,
            },
            "configuration": {
                "max_concurrent_extractions": self.settings.max_concurrent_extractions,
                "cleanup_temp": self.settings.cleanup_temp,
                "output_directory": str(self.settings.output_directory),
                "targets_configured": len(self.config.targets),
            },
            "file_system_summary": file_summary,
            "extraction_results": self.extraction_results,
            "failed_extractions": [
                {
                    "target": r["target"],
                    "image": r["image"],
                    "platform": r["platform"],
                    "errors": r.get("errors", []),
                }
                for r in failed_extractions
            ],
        }

        return summary

    def save_summary_report(self, filename: str = "extraction_report.json") -> Path:
        """
        Save summary report to file.

        Args:
            filename: Output filename

        Returns:
            Path to saved report
        """
        summary = self.generate_summary_report()
        report_path = self.settings.output_directory / filename

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Summary report saved to: {report_path}")
        return report_path

    def print_summary(self) -> None:
        """Print human-readable summary to console."""
        if not self.extraction_results:
            print("No extraction results available.")
            return

        summary = self.generate_summary_report()
        ext_summary = summary["extraction_summary"]

        print("\n" + "=" * 60)
        print("🎯 BINARY EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total Targets: {ext_summary['total_targets']}")
        print(f"Extraction Tasks: {ext_summary['total_tasks']}")
        print(f"Successful: {ext_summary['successful_extractions']} ✅")
        print(f"Failed: {ext_summary['failed_extractions']} ❌")
        print(f"Total Executables Found: {ext_summary['total_executables_found']:,}")
        print(f"Total Time: {ext_summary['total_extraction_time_seconds']:.1f}s")
        print(f"Average Time per Task: {ext_summary['average_extraction_time']:.1f}s")

        if summary.get("failed_extractions"):
            print("\n❌ Failed Extractions:")
            for failure in summary["failed_extractions"]:
                print(f"  • {failure['image']} ({failure['platform']})")
                for error in failure.get("errors", []):
                    print(f"    - {error}")

        fs_summary = summary.get("file_system_summary", {})
        if fs_summary:
            print("\n📊 File System Summary:")
            print(f"Distributions: {fs_summary.get('total_distributions', 0)}")
            print(f"Architectures: {fs_summary.get('total_architectures', 0)}")
            print(f"Total Size: {fs_summary.get('total_size_bytes', 0):,} bytes")

        print("=" * 60)
