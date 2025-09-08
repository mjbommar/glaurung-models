"""CLI entry point for binary extractor."""

import sys
from pathlib import Path
from typing import Optional

import click

from .config import ExtractionConfig
from .extractor import BinaryExtractor


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="extraction_config.json",
    help="Configuration file path (default: extraction_config.json)",
)
@click.option(
    "--max-workers",
    "-w",
    type=int,
    default=None,
    help="Maximum concurrent extractions (overrides config)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (overrides config)",
)
@click.option(
    "--temp-dir",
    "-t",
    type=click.Path(path_type=Path),
    default=None,
    help="Temporary directory (overrides config)",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Log level (overrides config)",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=None,
    help="Clean up temporary files (overrides config)",
)
@click.option(
    "--summary-only",
    "-s",
    is_flag=True,
    help="Generate summary report without extraction",
)
@click.option(
    "--validate-config", is_flag=True, help="Validate configuration file and exit"
)
def main(
    config: Path,
    max_workers: Optional[int],
    output_dir: Optional[Path],
    temp_dir: Optional[Path],
    log_level: Optional[str],
    cleanup: Optional[bool],
    summary_only: bool,
    validate_config: bool,
) -> None:
    """
    Modern Docker-based binary extraction system.

    Extracts system binaries from Docker images across multiple architectures
    using a clean filesystem copy approach that works consistently across
    all platforms.
    """
    try:
        # Load and validate configuration
        click.echo(f"📝 Loading configuration from {config}")
        extraction_config = ExtractionConfig.from_json(config)

        if validate_config:
            click.echo("✅ Configuration is valid!")
            _print_config_summary(extraction_config)
            return

        # Apply CLI overrides
        if output_dir:
            extraction_config.extraction_config.output_directory = output_dir
        if temp_dir:
            extraction_config.extraction_config.temp_directory = temp_dir
        if log_level:
            extraction_config.extraction_config.log_level = log_level.upper()
        if cleanup is not None:
            extraction_config.extraction_config.cleanup_temp = cleanup
        if max_workers:
            extraction_config.extraction_config.max_concurrent_extractions = max_workers

        # Initialize extractor
        extractor = BinaryExtractor(extraction_config)

        if summary_only:
            # Generate summary from existing extractions
            click.echo("📊 Generating summary report...")
            extractor.save_summary_report()
            extractor.print_summary()
            return

        # Print extraction plan
        _print_extraction_plan(extraction_config)

        # Run extraction
        click.echo("🔄 Starting binary extraction...")
        extractor.extract_all_targets(max_workers)

        # Generate and save reports
        click.echo("📝 Generating reports...")
        report_path = extractor.save_summary_report()

        # Print summary
        extractor.print_summary()

        click.echo("\n✅ Extraction complete!")
        click.echo(
            f"📁 Binaries saved to: {extraction_config.extraction_config.output_directory}"
        )
        click.echo(f"📊 Report saved to: {report_path}")

    except FileNotFoundError as e:
        click.echo(f"❌ Configuration file not found: {e}", err=True)
        sys.exit(1)

    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\n⚠️ Extraction interrupted by user", err=True)
        sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


def _print_config_summary(config: ExtractionConfig) -> None:
    """Print configuration summary."""
    click.echo("\n📋 Configuration Summary:")
    click.echo(f"  Output Directory: {config.extraction_config.output_directory}")
    click.echo(f"  Temp Directory: {config.extraction_config.temp_directory}")
    click.echo(f"  Log Level: {config.extraction_config.log_level}")
    click.echo(
        f"  Max Concurrent: {config.extraction_config.max_concurrent_extractions}"
    )
    click.echo(f"  Cleanup Temp: {config.extraction_config.cleanup_temp}")

    click.echo(f"\n🎯 Targets ({len(config.targets)}):")
    for target in config.targets:
        click.echo(f"  • {target.image} ({target.label})")
        click.echo(f"    Platforms: {', '.join(target.platforms)}")
        click.echo(f"    Paths: {len(target.paths)} patterns")


def _print_extraction_plan(config: ExtractionConfig) -> None:
    """Print extraction plan."""
    # Calculate total tasks
    total_tasks = sum(len(target.platforms) for target in config.targets)

    click.echo("\n🎯 Extraction Plan:")
    click.echo(f"  Targets: {len(config.targets)}")
    click.echo(f"  Total Tasks: {total_tasks}")
    click.echo(
        f"  Concurrent Workers: {config.extraction_config.max_concurrent_extractions}"
    )
    click.echo(f"  Output: {config.extraction_config.output_directory}")

    # Show platforms
    all_platforms = config.get_all_platforms()
    click.echo(f"  Platforms: {', '.join(sorted(all_platforms))}")


# Alternative entry points for different use cases
@click.command()
@click.argument("config_file", type=click.Path(path_type=Path))
def validate(config_file: Path) -> None:
    """Validate a configuration file."""
    try:
        config = ExtractionConfig.from_json(config_file)
        click.echo("✅ Configuration is valid!")
        _print_config_summary(config)
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)


@click.group()
def cli() -> None:
    """Binary Extractor CLI."""
    pass


# Add commands to group
cli.add_command(main, "extract")
cli.add_command(validate, "validate")


if __name__ == "__main__":
    main()
