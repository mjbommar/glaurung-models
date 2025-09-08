#!/bin/bash

# Simple analysis script for testing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARIES_DIR="$PROJECT_DIR/binaries"
LOGS_DIR="$PROJECT_DIR/logs"

# Create logs directory
mkdir -p "$LOGS_DIR"

echo "=== Binary Analysis Test ==="
echo "Project Directory: $PROJECT_DIR"
echo "Binaries Directory: $BINARIES_DIR"
echo ""

# Check if binaries exist
if [ ! -d "$BINARIES_DIR" ]; then
    echo "ERROR: No binaries directory found"
    exit 1
fi

# Analyze each directory
find "$BINARIES_DIR" -mindepth 2 -maxdepth 2 -type d | while IFS= read -r dir; do
    # Extract os_label and arch from path
    path_parts="${dir#$BINARIES_DIR/}"
    os_label="${path_parts%/*}"
    arch="${path_parts#*/}"
    
    echo "=== Analyzing: $os_label/$arch ==="
    echo "Directory: $dir"
    
    # Count files
    total_files=$(find "$dir" -type f 2>/dev/null | wc -l)
    exec_files=$(find "$dir" -type f -executable 2>/dev/null | wc -l)
    total_size=$(du -sb "$dir" 2>/dev/null | cut -f1)
    
    echo "  Total files: $total_files"
    echo "  Executable files: $exec_files"
    echo "  Total size: $total_size bytes"
    
    # List some sample files
    echo "  Sample files:"
    find "$dir" -type f -executable 2>/dev/null | head -5 | while read -r file; do
        basename_file=$(basename "$file")
        size=$(stat -c%s "$file" 2>/dev/null || echo "0")
        echo "    - $basename_file ($size bytes)"
    done
    
    echo ""
done

echo "=== Analysis Complete ==="