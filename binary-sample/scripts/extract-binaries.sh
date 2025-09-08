#!/bin/bash

# Binary Extraction from Docker Images
# This script extracts system binaries from various Docker images across different architectures
# for the purpose of creating a diverse sample dataset of normal system binaries.

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARIES_DIR="$PROJECT_DIR/binaries"
LOGS_DIR="$PROJECT_DIR/logs"
DOCKER_DIR="$PROJECT_DIR/docker"

# Common binary paths to extract
BINARY_PATHS=(
    "/usr/bin"
    "/bin"
    "/sbin"
    "/usr/sbin"
    "/usr/local/bin"
    "/usr/local/sbin"
)

# Image configurations: "image_name:tag:os_label:architectures"
IMAGE_CONFIGS=(
    "alpine:3.19:alpine3.19:linux/amd64,linux/arm64,linux/arm/v7"
    "alpine:3.18:alpine3.18:linux/amd64,linux/arm64,linux/arm/v7"
    "alpine:latest:alpine-latest:linux/amd64,linux/arm64,linux/arm/v7"
    "ubuntu:24.04:ubuntu24.04:linux/amd64,linux/arm64,linux/arm/v7"
    "ubuntu:22.04:ubuntu22.04:linux/amd64,linux/arm64,linux/arm/v7"
    "ubuntu:20.04:ubuntu20.04:linux/amd64,linux/arm64,linux/arm/v7"
    "debian:bookworm:debian-bookworm:linux/amd64,linux/arm64,linux/arm/v7"
    "debian:bullseye:debian-bullseye:linux/amd64,linux/arm64,linux/arm/v7"
    "rockylinux:9:rockylinux9:linux/amd64,linux/arm64"
    "almalinux:9:almalinux9:linux/amd64,linux/arm64"
)

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/extraction.log"
}

# Function to convert platform string to directory name
platform_to_dir() {
    local platform="$1"
    case "$platform" in
        "linux/amd64") echo "amd64" ;;
        "linux/arm64") echo "arm64" ;;
        "linux/arm/v7") echo "arm32v7" ;;
        *) echo "unknown" ;;
    esac
}

# Function to extract binaries from a container
extract_binaries() {
    local image="$1"
    local platform="$2"
    local os_label="$3"
    
    local arch_dir=$(platform_to_dir "$platform")
    local output_dir="$BINARIES_DIR/$os_label/$arch_dir"
    
    log "Extracting binaries from $image ($platform) to $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Pull the specific platform image
    log "Pulling $image for $platform"
    if ! docker pull --platform "$platform" "$image"; then
        log "ERROR: Failed to pull $image for $platform"
        return 1
    fi
    
    # Create and start container
    local container_name="binary-extractor-$(echo "$os_label-$arch_dir" | tr '.' '-')-$$"
    log "Creating container $container_name"
    
    if ! docker create --platform "$platform" --name "$container_name" "$image" sleep 3600; then
        log "ERROR: Failed to create container $container_name"
        return 1
    fi
    
    # Extract binaries from each path
    for binary_path in "${BINARY_PATHS[@]}"; do
        log "Extracting from $binary_path"
        
        # Check if path exists in container
        if docker run --platform "$platform" --rm "$image" test -d "$binary_path" 2>/dev/null; then
            # Create subdirectory for this path
            local path_name=$(basename "$binary_path")
            if [ "$path_name" = "bin" ]; then
                path_name="bin-$(dirname "$binary_path" | tr '/' '-' | sed 's/^-//')"
            fi
            
            local target_dir="$output_dir/$path_name"
            mkdir -p "$target_dir"
            
            # Copy files from container
            if docker cp "$container_name:$binary_path/." "$target_dir/" 2>/dev/null; then
                local count=$(find "$target_dir" -type f -executable 2>/dev/null | wc -l)
                log "Extracted $count executable files from $binary_path"
            else
                log "WARNING: Could not extract from $binary_path (may be empty or inaccessible)"
            fi
        else
            log "WARNING: Path $binary_path does not exist in $image"
        fi
    done
    
    # Clean up container
    docker rm "$container_name" >/dev/null 2>&1 || true
    
    # Create metadata file
    cat > "$output_dir/metadata.json" << EOF
{
    "image": "$image",
    "platform": "$platform",
    "architecture": "$arch_dir",
    "os_label": "$os_label",
    "extracted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "binary_paths": $(printf '%s\n' "${BINARY_PATHS[@]}" | jq -R . | jq -s .)
}
EOF
    
    log "Completed extraction for $image ($platform)"
}

# Function to process a single image configuration
process_image_config() {
    local config="$1"
    
    # Parse configuration: image:tag:os_label:architectures
    IFS=':' read -r image_name tag os_label architectures <<< "$config"
    local full_image="$image_name:$tag"
    
    log "Processing $full_image ($os_label)"
    
    # Split architectures by comma
    IFS=',' read -ra ARCHS <<< "$architectures"
    
    for arch in "${ARCHS[@]}"; do
        extract_binaries "$full_image" "$arch" "$os_label"
    done
}

# Main execution function
main() {
    log "Starting binary extraction process"
    log "Project directory: $PROJECT_DIR"
    log "Output directory: $BINARIES_DIR"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log "ERROR: Docker is not available"
        exit 1
    fi
    
    # Check Docker daemon is running
    if ! docker info &> /dev/null; then
        log "ERROR: Docker daemon is not running"
        exit 1
    fi
    
    # Enable Docker buildx for multi-platform support
    docker buildx create --use --name multiarch-builder --driver docker-container >/dev/null 2>&1 || true
    
    # Process each image configuration
    local total_configs=${#IMAGE_CONFIGS[@]}
    local current=0
    
    for config in "${IMAGE_CONFIGS[@]}"; do
        current=$((current + 1))
        log "Processing configuration $current/$total_configs: $config"
        
        if ! process_image_config "$config"; then
            log "ERROR: Failed to process configuration: $config"
        fi
        
        log "Completed $current/$total_configs configurations"
    done
    
    # Generate summary
    log "Generating summary report"
    find "$BINARIES_DIR" -name "*.json" -exec cat {} \; | jq -s . > "$LOGS_DIR/extraction-summary.json"
    
    log "Binary extraction process completed"
    log "Summary report saved to: $LOGS_DIR/extraction-summary.json"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help] [--dry-run]"
        echo "Extract system binaries from Docker images across multiple architectures"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --dry-run     Show what would be done without actually doing it"
        exit 0
        ;;
    --dry-run)
        log "DRY RUN MODE: Would process ${#IMAGE_CONFIGS[@]} image configurations"
        for config in "${IMAGE_CONFIGS[@]}"; do
            echo "Would process: $config"
        done
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Use --help for usage information" >&2
        exit 1
        ;;
esac