#!/bin/bash

# Binary Collection Orchestration Script
# Runs the complete binary collection and analysis process

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs"

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/collection.log"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR: Docker is not installed"
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        log "ERROR: Docker daemon is not running"
        return 1
    fi
    
    # Check jq for JSON processing
    if ! command -v jq &> /dev/null; then
        log "WARNING: jq is not installed. Some analysis features may not work properly."
        log "To install jq: sudo apt-get install jq (Ubuntu/Debian) or brew install jq (macOS)"
    fi
    
    # Check available disk space
    local available_space=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    log "Available disk space: ${available_gb}GB"
    if [ "$available_gb" -lt 5 ]; then
        log "WARNING: Less than 5GB available space. Binary extraction may require significant disk space."
    fi
    
    log "Prerequisites check completed"
}

# Function to run extraction
run_extraction() {
    log "Starting binary extraction process"
    
    if [ -x "$SCRIPT_DIR/extract-binaries.sh" ]; then
        "$SCRIPT_DIR/extract-binaries.sh"
    else
        log "ERROR: extract-binaries.sh not found or not executable"
        return 1
    fi
    
    log "Binary extraction completed"
}

# Function to run analysis
run_analysis() {
    log "Starting binary analysis process"
    
    if [ -x "$SCRIPT_DIR/analyze-binaries.sh" ]; then
        "$SCRIPT_DIR/analyze-binaries.sh"
    else
        log "ERROR: analyze-binaries.sh not found or not executable"
        return 1
    fi
    
    log "Binary analysis completed"
}

# Function to generate final report
generate_final_report() {
    log "Generating final collection report"
    
    local report_file="$PROJECT_DIR/COLLECTION-REPORT.md"
    
    cat > "$report_file" << 'EOF'
# Binary Collection Report

This report summarizes the system binary collection process across multiple Docker images and architectures.

## Collection Summary

EOF
    
    # Add timestamp
    echo "**Generated:** $(date)" >> "$report_file"
    echo "" >> "$report_file"
    
    # Add directory structure
    echo "## Directory Structure" >> "$report_file"
    echo "" >> "$report_file"
    echo '```' >> "$report_file"
    find "$PROJECT_DIR" -type d | head -20 >> "$report_file"
    echo '```' >> "$report_file"
    echo "" >> "$report_file"
    
    # Add statistics if available
    if [ -f "$LOGS_DIR/binary-analysis-summary.json" ]; then
        echo "## Collection Statistics" >> "$report_file"
        echo "" >> "$report_file"
        
        if command -v jq &> /dev/null; then
            local total_files=$(jq -r '.overall_summary.total_files' "$LOGS_DIR/binary-analysis-summary.json" 2>/dev/null || echo "unknown")
            local total_executable=$(jq -r '.overall_summary.total_executable_files' "$LOGS_DIR/binary-analysis-summary.json" 2>/dev/null || echo "unknown")
            local total_size=$(jq -r '.overall_summary.total_size_bytes' "$LOGS_DIR/binary-analysis-summary.json" 2>/dev/null || echo "unknown")
            local distributions=$(jq -r '.overall_summary.distributions_analyzed' "$LOGS_DIR/binary-analysis-summary.json" 2>/dev/null || echo "unknown")
            
            cat >> "$report_file" << EOF
- **Total Files Collected:** $total_files
- **Executable Files:** $total_executable
- **Total Size:** $total_size bytes
- **Distributions Analyzed:** $distributions

EOF
        fi
    fi
    
    # Add log files section
    echo "## Log Files" >> "$report_file"
    echo "" >> "$report_file"
    find "$LOGS_DIR" -name "*.log" -o -name "*.json" | while read -r logfile; do
        echo "- $(basename "$logfile")" >> "$report_file"
    done
    
    echo "" >> "$report_file"
    echo "## Usage" >> "$report_file"
    cat >> "$report_file" << 'EOF'

To use the collected binaries:

1. Navigate to the `binaries/` directory
2. Choose the desired OS distribution and architecture
3. Binary files are organized by their original system path (e.g., `usr-bin/`, `bin/`)

## Scripts

- `scripts/extract-binaries.sh` - Main extraction script
- `scripts/analyze-binaries.sh` - Analysis and reporting script  
- `scripts/run-collection.sh` - Complete orchestration script

For more details, see the log files in the `logs/` directory.
EOF
    
    log "Final report generated: $report_file"
}

# Function to cleanup Docker resources
cleanup_docker() {
    log "Cleaning up Docker resources"
    
    # Remove any leftover containers
    docker ps -a --filter "name=binary-extractor*" --format "{{.Names}}" | while read -r container; do
        if [ -n "$container" ]; then
            log "Removing container: $container"
            docker rm -f "$container" >/dev/null 2>&1 || true
        fi
    done
    
    # Clean up buildx builder if it exists
    docker buildx rm multiarch-builder >/dev/null 2>&1 || true
    
    log "Docker cleanup completed"
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    log "=== Binary Collection Process Started ==="
    log "Project directory: $PROJECT_DIR"
    
    # Run prerequisite checks
    if ! check_prerequisites; then
        log "ERROR: Prerequisites check failed"
        exit 1
    fi
    
    # Run extraction
    if ! run_extraction; then
        log "ERROR: Binary extraction failed"
        cleanup_docker
        exit 1
    fi
    
    # Run analysis
    if ! run_analysis; then
        log "ERROR: Binary analysis failed"
        cleanup_docker
        exit 1
    fi
    
    # Generate final report
    generate_final_report
    
    # Cleanup
    cleanup_docker
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "=== Binary Collection Process Completed ==="
    log "Total duration: ${duration} seconds"
    log "Check COLLECTION-REPORT.md for summary"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help] [--extract-only] [--analyze-only]"
        echo "Run the complete binary collection and analysis process"
        echo ""
        echo "Options:"
        echo "  --help, -h        Show this help message"
        echo "  --extract-only    Run only the extraction process"
        echo "  --analyze-only    Run only the analysis process"
        echo "  --cleanup         Clean up Docker resources only"
        exit 0
        ;;
    --extract-only)
        log "Running extraction only"
        check_prerequisites && run_extraction
        ;;
    --analyze-only)
        log "Running analysis only"
        run_analysis && generate_final_report
        ;;
    --cleanup)
        cleanup_docker
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