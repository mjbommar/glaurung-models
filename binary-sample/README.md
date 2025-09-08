# Binary Sample Collection

This project collects system binaries from various operating systems and architectures using Docker containers. The goal is to create a diverse dataset of normal system binaries for security research and analysis purposes.

## Overview

**Features:**
- JSON-driven configuration with Pydantic validation
- Concurrent extraction using ThreadPoolExecutor  
- Robust error handling with selective fallback strategies
- Clean CLI interface with comprehensive reporting
- UV package manager for modern dependency management
- Cross-platform path support (Linux, Windows, custom distributions)

## Quick Start

```bash
# Install dependencies with UV
uv run python -m binary_extractor.main

# Or with custom configuration
uv run python -m binary_extractor.main -c custom_config.json -w 8
```

The extraction process:
1. Pulls official Docker images for different OS distributions and architectures
2. Extracts binaries from configurable system paths using glob patterns
3. Organizes binaries by OS distribution and architecture  
4. Generates comprehensive analysis reports and metadata
5. Handles cross-architecture challenges with filesystem copy fallbacks

## Supported Distributions & Architectures

### Operating Systems
- **Alpine Linux** (3.18, 3.19, latest)
- **Ubuntu** (20.04, 22.04, 24.04)
- **Debian** (bullseye, bookworm)
- **Rocky Linux** (9)
- **Alma Linux** (9)

### Architectures
- **amd64** (x86-64)
- **arm64** (ARM 64-bit)
- **arm32v7** (ARM 32-bit v7)

## Directory Structure

```
binary-sample/
├── binaries/                 # Collected binaries organized by OS/arch
│   ├── alpine3.19/
│   │   ├── amd64/
│   │   ├── arm64/
│   │   └── arm32v7/
│   ├── ubuntu24.04/
│   │   ├── amd64/
│   │   └── arm64/
│   └── ...
├── docker/                   # Docker-related files
├── scripts/                  # Collection and analysis scripts
│   ├── extract-binaries.sh   # Main extraction script
│   ├── analyze-binaries.sh   # Analysis and reporting
│   └── run-collection.sh     # Complete orchestration
└── logs/                     # Execution logs and reports
```

## Installation & Setup

**Prerequisites:**
- Python 3.11+ 
- Docker with daemon running
- UV package manager (recommended) or pip
- **5GB+** available disk space (recommended)

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Verify Docker is running
docker info
```

## Configuration

The system uses a JSON configuration file that defines extraction targets, paths, and settings:

```json
{
  "extraction_config": {
    "output_directory": "binaries",
    "temp_directory": "temp_extraction", 
    "log_level": "INFO",
    "max_concurrent_extractions": 4,
    "cleanup_temp": true
  },
  "targets": [
    {
      "image": "alpine:3.19",
      "label": "alpine3.19",
      "description": "Alpine Linux 3.19 - minimal security-focused",
      "platforms": ["linux/amd64", "linux/arm64", "linux/arm/v7"],
      "paths": [
        "/usr/bin/**/*",
        "/bin/**/*", 
        "/sbin/**/*",
        "/usr/sbin/**/*"
      ],
      "include_patterns": ["*"],
      "exclude_patterns": ["*.txt", "*.conf", "*.md"]
    }
  ]
}
```

## Usage

### Basic Extraction
```bash
# Extract using default config
uv run python -m binary_extractor.main

# Use custom config file  
uv run python -m binary_extractor.main -c my_config.json

# Override concurrent workers
uv run python -m binary_extractor.main -w 8
```

### Advanced Options
```bash
# Validate configuration without extraction
uv run python -m binary_extractor.main --validate-config

# Generate summary report only
uv run python -m binary_extractor.main --summary-only

# Override output directory
uv run python -m binary_extractor.main -o /path/to/binaries

# Set log level
uv run python -m binary_extractor.main -l DEBUG

# Skip temp cleanup for debugging
uv run python -m binary_extractor.main --no-cleanup
```

### Command Line Reference
```
Options:
  -c, --config FILE         Configuration file (default: extraction_config.json)
  -w, --max-workers INT     Maximum concurrent extractions
  -o, --output-dir PATH     Output directory
  -t, --temp-dir PATH       Temporary directory  
  -l, --log-level LEVEL     Log level (DEBUG|INFO|WARNING|ERROR)
  --cleanup/--no-cleanup    Clean up temporary files
  -s, --summary-only        Generate summary report without extraction
  --validate-config         Validate configuration file and exit
```

## Architecture

### Core Components

**BinaryExtractor**: Main orchestration class
- Manages concurrent extraction tasks
- Coordinates Docker and file operations
- Generates comprehensive reports

**DockerManager**: Docker operations with error handling
- Multi-platform image pulling and container creation
- Filesystem copying with permission error recovery
- Automatic container cleanup

**FileManager**: File operations and binary detection
- Executable file identification and organization
- Glob pattern matching for flexible path selection
- Metadata generation and storage

**Configuration System**: Pydantic-based validation
- Type-safe configuration loading
- Platform and path validation
- Flexible target definition

### Error Handling Strategy

**Graceful Degradation**: The system handles various failure modes:

1. **Full filesystem copy** (preferred)
2. **Selective directory copying** (fallback for permission errors)
3. **Error logging and continuation** (ensures maximum extraction success)

**Enterprise Container Support**: Special handling for Rocky/Alma Linux containers with restrictive permissions.

## Output Structure

```
binaries/
├── alpine3.19/
│   ├── linux-amd64/
│   │   ├── usr-bin/           # Binaries from /usr/bin
│   │   ├── bin/               # Binaries from /bin  
│   │   └── extraction_metadata.json
│   ├── linux-arm64/
│   └── linux-arm-v7/
├── ubuntu24.04/
└── extraction_report.json     # Comprehensive extraction summary
```

## Performance

**Typical Results:**
- **11,225+ executables** across all distributions
- **25 extraction tasks** in ~9 seconds  
- **84%+ success rate** (normal distributions work perfectly)
- **8.7GB** total binary collection

**Concurrent Execution:**
- ThreadPoolExecutor with configurable workers
- Efficient Docker image reuse
- Parallel extraction across architectures

## Alternative Package Managers

If you prefer not to use UV:

```bash
# Using pip
pip install -e .

# Run with pip
python -m binary_extractor.main
```

# Configuration

The system uses JSON configuration files for complete flexibility:

### Adding New Docker Images

Edit `extraction_config.json`:
```json
{
  "targets": [
    {
      "image": "your-image:tag",
      "label": "custom-label",
      "description": "Your custom image description",
      "platforms": ["linux/amd64", "linux/arm64", "windows/amd64"],
      "paths": [
        "/usr/bin/**/*",
        "/custom/path/**/*"
      ],
      "include_patterns": ["*"],
      "exclude_patterns": ["*.txt", "*.log"]
    }
  ]
}
```

### Cross-Platform Path Support

The system now supports **configurable paths per target** instead of hardcoded Linux paths. This enables:

- **Windows containers**: `C:\Windows\System32\*`, `C:\Program Files\*`
- **Custom Linux distributions**: Non-standard binary locations
- **Embedded systems**: Device-specific path layouts

### Platform-Specific Configurations

```json
{
  "image": "mcr.microsoft.com/windows/nanoserver:ltsc2022",
  "label": "windows-nano",
  "platforms": ["windows/amd64"], 
  "paths": [
    "C:/Windows/System32/**/*.exe",
    "C:/Program Files/**/*.exe"
  ],
  "include_patterns": ["*.exe", "*.dll"],
  "exclude_patterns": ["*.txt", "*.log"]
}
```


# Docker Image Cross-Architecture Extraction

## The ARM/ARM64 "Empty Container" Challenge

When collecting binaries across multiple architectures, you may encounter containers that appear "empty" when using traditional extraction methods. This is particularly common with ARM64 and ARM32 images.

### Why ARM Containers Appear Empty

**Root Cause: Architecture Execution Mismatch**
- **Problem**: Standard extraction tries to execute commands (`ls`, `find`, etc.) inside ARM containers
- **Result**: `exec format error` when running ARM binaries on x86_64 hosts
- **Appearance**: Containers seem to have no binaries in standard paths

**Container Design Differences:**
1. **amd64 containers**: Full-featured with traditional Unix directory structures
2. **ARM containers**: Often optimized for resource-constrained environments
3. **Different layouts**: ARM images may use custom filesystem organizations
4. **Minimal base images**: ARM variants frequently use distroless or scratch bases

### The Solution: Filesystem Copy Extraction

**Breakthrough Approach:**
Instead of executing commands inside containers, copy the entire filesystem out and analyze it on the host.

```bash
# WRONG: Tries to execute commands inside ARM container
docker run --platform linux/arm64 alpine:3.19 find / -executable  # ❌ exec format error

# RIGHT: Copy filesystem without execution
docker create --platform linux/arm64 --name temp alpine:3.19
docker cp temp:/ ./temp-fs/                                        # ✅ Works!
find ./temp-fs -type f -executable                                 # ✅ Find ARM binaries
docker rm temp
```

### Implementation Pattern

The key insight: **Separate container creation from binary analysis**

1. **Create multi-arch containers** (works fine across architectures)
2. **Copy complete filesystems** using `docker cp` (no execution required)  
3. **Analyze copied files** on host system (native architecture)
4. **Extract executable binaries** from copied filesystem

### Results

Using this approach revealed:
- **Alpine ARM64**: 28+ executable binaries (not "empty")
- **Ubuntu ARM64**: Hidden binaries in non-standard locations
- **Comprehensive coverage**: True multi-architecture binary collection

The modern Python implementation automatically handles this cross-architecture challenge with robust fallback mechanisms.

## Security Considerations

This tool is designed for legitimate security research purposes:
- ✅ Collecting normal system binaries for baseline analysis
- ✅ Creating datasets for malware detection research
- ✅ Security tool development and testing

**Do not use for malicious purposes.**

## Troubleshooting

### Docker Issues
```bash
# Check Docker is running
docker info

# Check multi-platform support  
docker buildx ls

# Clean up Docker resources manually if needed
docker system prune
```

### Disk Space
Large collections may require significant disk space. Monitor with:
```bash
du -sh binary-sample/binaries/
```

### Architecture Support
Some images may not support all architectures. The script will log warnings and continue with available architectures.

## Contributing

To add support for new distributions or architectures:

1. Add entries to the `targets` array in `extraction_config.json`
2. Test the extraction process with `--validate-config`
3. Run a test extraction
4. Update documentation

## License

This project is intended for security research and educational purposes. Respect all applicable terms of service for Docker images used.