#!/bin/bash

# Test script to validate extraction process with a single image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARIES_DIR="$PROJECT_DIR/binaries"
LOGS_DIR="$PROJECT_DIR/logs"

# Create logs directory
mkdir -p "$LOGS_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/test-extraction.log"
}

# Test with Alpine Linux (small and fast)
TEST_IMAGE="alpine:3.19"
TEST_PLATFORM="linux/amd64"
TEST_OS_LABEL="alpine3.19"
TEST_ARCH="amd64"

BINARY_PATHS=(
    "/usr/bin"
    "/bin"
    "/sbin"
)

log "Starting test extraction"
log "Image: $TEST_IMAGE"
log "Platform: $TEST_PLATFORM"

# Check Docker
if ! docker info &> /dev/null; then
    log "ERROR: Docker daemon is not running"
    exit 1
fi

# Pull image
log "Pulling $TEST_IMAGE for $TEST_PLATFORM"
if ! docker pull --platform "$TEST_PLATFORM" "$TEST_IMAGE"; then
    log "ERROR: Failed to pull image"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$BINARIES_DIR/$TEST_OS_LABEL/$TEST_ARCH"
mkdir -p "$OUTPUT_DIR"

# Create container
CONTAINER_NAME="test-extractor-$$"
log "Creating test container: $CONTAINER_NAME"

if ! docker create --platform "$TEST_PLATFORM" --name "$CONTAINER_NAME" "$TEST_IMAGE" sleep 3600; then
    log "ERROR: Failed to create container"
    exit 1
fi

# Extract binaries
for binary_path in "${BINARY_PATHS[@]}"; do
    log "Testing extraction from $binary_path"
    
    if docker run --platform "$TEST_PLATFORM" --rm "$TEST_IMAGE" test -d "$binary_path" 2>/dev/null; then
        path_name=$(basename "$binary_path")
        if [ "$path_name" = "bin" ]; then
            path_name="bin-$(dirname "$binary_path" | tr '/' '-' | sed 's/^-//')"
        fi
        
        target_dir="$OUTPUT_DIR/$path_name"
        mkdir -p "$target_dir"
        
        if docker cp "$CONTAINER_NAME:$binary_path/." "$target_dir/" 2>/dev/null; then
            count=$(find "$target_dir" -type f 2>/dev/null | wc -l)
            exec_count=$(find "$target_dir" -type f -executable 2>/dev/null | wc -l)
            log "SUCCESS: Extracted $count files ($exec_count executable) from $binary_path"
        else
            log "WARNING: Could not extract from $binary_path"
        fi
    else
        log "WARNING: Path $binary_path does not exist"
    fi
done

# List some extracted files
log "Sample of extracted files:"
find "$OUTPUT_DIR" -type f -executable 2>/dev/null | head -10 | while read -r file; do
    log "  - $(basename "$file")"
done

# Create test metadata
cat > "$OUTPUT_DIR/test-metadata.json" << EOF
{
    "test_extraction": true,
    "image": "$TEST_IMAGE",
    "platform": "$TEST_PLATFORM",
    "extracted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# Cleanup
docker rm "$CONTAINER_NAME" >/dev/null 2>&1

# Final statistics
total_files=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l)
exec_files=$(find "$OUTPUT_DIR" -type f -executable 2>/dev/null | wc -l)
total_size=$(du -sb "$OUTPUT_DIR" 2>/dev/null | cut -f1)

log "Test extraction completed successfully!"
log "Total files: $total_files"
log "Executable files: $exec_files"  
log "Total size: $total_size bytes"
log "Output directory: $OUTPUT_DIR"