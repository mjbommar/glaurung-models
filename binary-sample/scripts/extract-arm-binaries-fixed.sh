#!/bin/bash

# FIXED ARM Binary Extraction - Copy entire filesystem without execution
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARIES_DIR="$PROJECT_DIR/binaries"
LOGS_DIR="$PROJECT_DIR/logs"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/arm-extraction-fixed.log"
}

# ARM containers that appeared "empty"
ARM_CONFIGS=(
    "alpine:3.19:alpine3.19:linux/arm64"
    "alpine:3.18:alpine3.18:linux/arm64"
    "ubuntu:22.04:ubuntu22.04:linux/arm64"
    "ubuntu:20.04:ubuntu20.04:linux/arm64"
    "debian:bookworm:debian-bookworm:linux/arm64"
)

# Function to extract ARM binaries without execution
extract_arm_filesystem() {
    local image="$1"
    local platform="$2"
    local os_label="$3"
    
    local arch_dir="arm64"
    local output_dir="$BINARIES_DIR/$os_label/$arch_dir"
    local temp_dir="$output_dir/temp-fs-extract"
    
    log "FIXED: Extracting complete filesystem from $image ($platform)"
    
    # Create container
    local container_name="arm-extractor-$(date +%s)-$$"
    log "Creating container: $container_name"
    
    if ! docker create --platform "$platform" --name "$container_name" "$image" sleep 3600; then
        log "ERROR: Failed to create container $container_name"
        return 1
    fi
    
    # Create temp directory for filesystem extraction
    mkdir -p "$temp_dir"
    
    # Copy ENTIRE filesystem
    log "Copying entire container filesystem..."
    if docker cp "$container_name:/" "$temp_dir/" 2>/dev/null; then
        log "Successfully copied filesystem"
        
        # Find ALL executable files in the extracted filesystem
        log "Scanning for executable files..."
        find "$temp_dir" -type f -executable 2>/dev/null > "$temp_dir/executables.list" || true
        
        # Count and organize executables
        local exec_count=$(wc -l < "$temp_dir/executables.list" 2>/dev/null || echo "0")
        log "Found $exec_count executable files"
        
        # Create organized directory structure
        mkdir -p "$output_dir/arm64-executables"
        
        # Copy executables to organized structure
        local copied_count=0
        while IFS= read -r exec_file; do
            if [ -f "$exec_file" ]; then
                local rel_path="${exec_file#$temp_dir/}"
                local dest_dir="$output_dir/arm64-executables/$(dirname "$rel_path")"
                mkdir -p "$dest_dir"
                
                if cp "$exec_file" "$dest_dir/" 2>/dev/null; then
                    ((copied_count++))
                fi
            fi
        done < "$temp_dir/executables.list"
        
        log "Successfully copied $copied_count executable files"
        
        # Create summary
        cat > "$output_dir/arm64-extraction-summary.json" << EOF
{
    "image": "$image",
    "platform": "$platform", 
    "extraction_method": "full_filesystem_copy",
    "extracted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_executables": $copied_count,
    "extraction_success": true,
    "temp_directory": "$temp_dir"
}
EOF
        
        # Clean up temp directory
        rm -rf "$temp_dir"
        
    else
        log "WARNING: Could not copy filesystem from $container_name"
    fi
    
    # Cleanup container
    docker rm "$container_name" >/dev/null 2>&1 || true
    
    log "Completed ARM extraction for $image ($platform)"
}

# Main execution
log "Starting FIXED ARM binary extraction"

for config in "${ARM_CONFIGS[@]}"; do
    IFS=':' read -r image tag os_label platform <<< "$config"
    extract_arm_filesystem "$image:$tag" "$platform" "$os_label"
done

log "FIXED ARM extraction completed"