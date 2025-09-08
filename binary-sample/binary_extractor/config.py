"""Configuration models with Pydantic validation."""

import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ExtractionSettings(BaseModel):
    """Global extraction settings."""

    output_directory: Path = Field(
        default="binaries", description="Output directory for extracted binaries"
    )
    temp_directory: Path = Field(
        default="temp_extraction", description="Temporary directory for extraction"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    max_concurrent_extractions: int = Field(
        default=4, ge=1, le=20, description="Maximum concurrent extractions"
    )
    cleanup_temp: bool = Field(
        default=True, description="Clean up temporary files after extraction"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class Target(BaseModel):
    """Docker image target configuration."""

    image: str = Field(..., description="Docker image name with tag")
    label: str = Field(..., description="Unique label for organizing output")
    description: Optional[str] = Field(None, description="Human-readable description")
    platforms: List[str] = Field(
        ..., min_length=1, description="List of platforms to extract"
    )
    paths: List[str] = Field(
        ..., min_length=1, description="List of glob patterns for paths to extract"
    )
    include_patterns: List[str] = Field(
        default=["*"], description="Patterns for files to include"
    )
    exclude_patterns: List[str] = Field(
        default=[], description="Patterns for files to exclude"
    )

    @field_validator("platforms")
    @classmethod
    def validate_platforms(cls, v: List[str]) -> List[str]:
        """Validate platform format."""
        valid_prefixes = ("linux/", "windows/", "darwin/")
        for platform in v:
            if not any(platform.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(
                    f"Platform '{platform}' should start with one of {valid_prefixes}"
                )
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validate Docker image format."""
        if not v or ":" not in v:
            raise ValueError("Image must be in format 'name:tag'")
        return v

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label format for filesystem safety."""
        if not v.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError(
                "Label must contain only alphanumeric characters, hyphens, underscores, and dots"
            )
        return v


class ExtractionConfig(BaseModel):
    """Complete extraction configuration."""

    extraction_config: ExtractionSettings = Field(
        ..., description="Global extraction settings"
    )
    targets: List[Target] = Field(
        ..., min_length=1, description="List of extraction targets"
    )

    @model_validator(mode="after")
    def validate_unique_labels(self) -> "ExtractionConfig":
        """Ensure all target labels are unique."""
        labels = [target.label for target in self.targets]
        if len(labels) != len(set(labels)):
            raise ValueError("All target labels must be unique")
        return self

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> "ExtractionConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

        return cls.model_validate(data)

    def to_json(self, config_path: Union[str, Path], indent: int = 2) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=indent, default=str)

    def get_all_platforms(self) -> set[str]:
        """Get all unique platforms across targets."""
        platforms = set()
        for target in self.targets:
            platforms.update(target.platforms)
        return platforms

    def get_targets_by_platform(self, platform: str) -> List[Target]:
        """Get all targets that support a specific platform."""
        return [target for target in self.targets if platform in target.platforms]
