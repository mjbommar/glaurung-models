#!/bin/bash

# Binary Analysis Script
# Analyzes extracted binaries and generates detailed reports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARIES_DIR="$PROJECT_DIR/binaries"
LOGS_DIR="$PROJECT_DIR/logs"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS_DIR/analysis.log"
}

# Function to get file information
get_file_info() {
    local file="$1"
    local output=""
    
    # Check if file is executable
    if [ -x "$file" ]; then
        output="executable"
    else
        output="not-executable"
    fi
    
    # Get file type
    if command -v file &> /dev/null; then
        local file_type=$(file -b "$file" 2>/dev/null || echo "unknown")
        output="$output|$file_type"
    fi
    
    # Get file size
    local size=$(stat -c%s "$file" 2>/dev/null || echo "0")
    output="$output|$size"
    
    echo "$output"
}

# Function to analyze binaries in a directory
analyze_directory() {
    local dir="$1"
    local os_label="$2"
    local arch="$3"
    
    log "Analyzing binaries in $dir ($os_label/$arch)"
    
    local output_file="$LOGS_DIR/analysis-$os_label-$arch.json"
    local total_files=0
    local executable_files=0
    local total_size=0
    
    # Create temporary file for JSON array
    local temp_file=$(mktemp)
    echo "[" > "$temp_file"
    
    local first=true
    while IFS= read -r -d '' file; do
        total_files=$((total_files + 1))
        
        local info=$(get_file_info "$file")
        IFS='|' read -r executable file_type size <<< "$info"
        
        if [ "$executable" = "executable" ]; then
            executable_files=$((executable_files + 1))
        fi
        
        total_size=$((total_size + size))
        
        # Add comma separator except for first entry
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$temp_file"
        fi
        
        # Create JSON entry
        local relative_path="${file#$dir/}"
        cat >> "$temp_file" << EOF
    {
        "path": "$relative_path",
        "full_path": "$file",
        "executable": $([ "$executable" = "executable" ] && echo "true" || echo "false"),
        "file_type": $(echo "$file_type" | jq -R .),
        "size": $size,
        "basename": "$(basename "$file")"
    }
    done < <(find "$dir" -type f -print0 2>/dev/null)
    
    echo "]" >> "$temp_file"
    
    # Create final JSON report
    cat > "$output_file" << EOF
{
    "os_label": "$os_label",
    "architecture": "$arch",
    "directory": "$dir",
    "analysis_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "summary": {
        "total_files": $total_files,
        "executable_files": $executable_files,
        "total_size_bytes": $total_size,
        "average_size_bytes": $([ $total_files -gt 0 ] && echo "$((total_size / total_files))" || echo "0")
    },
    "files": $(cat "$temp_file")
}
EOF
    
    rm "$temp_file"
    
    log "Analysis complete: $total_files files ($executable_files executable), ${total_size} bytes total"
}

# Function to generate summary report
generate_summary() {
    log "Generating overall summary report"
    
    local summary_file="$LOGS_DIR/binary-analysis-summary.json"
    local temp_file=$(mktemp)
    
    echo "{" > "$temp_file"
    echo "  \"analysis_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$temp_file"
    echo "  \"project_directory\": \"$PROJECT_DIR\"," >> "$temp_file"
    echo "  \"distributions\": [" >> "$temp_file"
    
    local first=true
    local total_files=0
    local total_executable=0
    local total_size=0
    
    # Process each analysis file
    for analysis_file in "$LOGS_DIR"/analysis-*.json; do
        if [ -f "$analysis_file" ]; then
            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> "$temp_file"
            fi
            
            # Extract summary info
            local os_arch=$(basename "$analysis_file" .json | sed 's/analysis-//')
            local files_count=$(jq '.summary.total_files' "$analysis_file" 2>/dev/null || echo "0")
            local exec_count=$(jq '.summary.executable_files' "$analysis_file" 2>/dev/null || echo "0")
            local size_bytes=$(jq '.summary.total_size_bytes' "$analysis_file" 2>/dev/null || echo "0")
            
            total_files=$((total_files + files_count))
            total_executable=$((total_executable + exec_count))
            total_size=$((total_size + size_bytes))
            
            cat >> "$temp_file" << EOF
    {
        "os_arch": "$os_arch",
        "total_files": $files_count,
        "executable_files": $exec_count,
        "total_size_bytes": $size_bytes,
        "analysis_file": "$(basename "$analysis_file")"
    }
        fi
    done
    
    echo "  ]," >> "$temp_file"
    cat >> "$temp_file" << EOF
  "overall_summary": {
    "total_files": $total_files,
    "total_executable_files": $total_executable,
    "total_size_bytes": $total_size,
    "distributions_analyzed": $(find "$LOGS_DIR" -name "analysis-*.json" | wc -l)
  }
}
EOF
    
    mv "$temp_file" "$summary_file"
    log "Summary report saved to: $summary_file"
}

# Function to find common binaries across distributions
find_common_binaries() {
    log "Finding common binaries across distributions"
    
    local common_file="$LOGS_DIR/common-binaries.json"
    local temp_file=$(mktemp)
    
    # Extract all binary names
    find "$BINARIES_DIR" -type f -executable -printf "%f\n" 2>/dev/null | sort | uniq -c | sort -nr > "$temp_file"
    
    # Convert to JSON
    echo "{" > "$common_file"
    echo "  \"analysis_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$common_file"
    echo "  \"common_binaries\": [" >> "$common_file"
    
    local first=true
    while read -r count name; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$common_file"
        fi
        
        echo "    {\"name\": \"$name\", \"count\": $count}" >> "$common_file"
    done < "$temp_file"
    
    echo "  ]" >> "$common_file"
    echo "}" >> "$common_file"
    
    rm "$temp_file"
    log "Common binaries report saved to: $common_file"
}

# Main function
main() {
    log "Starting binary analysis"
    
    # Check if binaries directory exists
    if [ ! -d "$BINARIES_DIR" ]; then
        log "ERROR: Binaries directory not found: $BINARIES_DIR"
        exit 1
    fi
    
    # Analyze each distribution/architecture combination
    find "$BINARIES_DIR" -mindepth 2 -maxdepth 2 -type d | while IFS= read -r dir; do
        # Extract os_label and arch from path
        local path_parts="${dir#$BINARIES_DIR/}"
        local os_label="${path_parts%/*}"
        local arch="${path_parts#*/}"
        
        analyze_directory "$dir" "$os_label" "$arch"
    done
    
    # Generate summary reports
    generate_summary
    find_common_binaries
    
    log "Binary analysis completed"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo "Analyze extracted system binaries and generate reports"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
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