#!/bin/bash

# Deep exploration of ARM container filesystem structures
set -euo pipefail

# Test a few ARM containers to see their actual structure
CONTAINERS=(
    "alpine:3.19:linux/arm64"
    "ubuntu:22.04:linux/arm64" 
    "debian:bookworm:linux/arm64"
)

echo "=== DEEP ARM CONTAINER EXPLORATION ==="

for container_config in "${CONTAINERS[@]}"; do
    IFS=':' read -r image tag platform <<< "$container_config"
    full_image="$image:$tag"
    
    echo ""
    echo "🔍 Exploring $full_image ($platform)"
    echo "=================================="
    
    # Create container
    container_name="explorer-$(date +%s)-$$"
    echo "Creating container: $container_name"
    
    if docker create --platform "$platform" --name "$container_name" "$full_image" sleep 3600; then
        
        echo "📁 ROOT DIRECTORY CONTENTS:"
        docker run --platform "$platform" --rm "$full_image" ls -la / 2>/dev/null || echo "  (ls failed)"
        
        echo ""
        echo "🔍 SEARCHING FOR EXECUTABLES EVERYWHERE:"
        docker run --platform "$platform" --rm "$full_image" sh -c 'find / -type f -executable 2>/dev/null | head -20' || echo "  (find failed)"
        
        echo ""
        echo "📋 ALL FILES IN ROOT:"
        docker run --platform "$platform" --rm "$full_image" sh -c 'find / -maxdepth 3 -type f 2>/dev/null | head -30' || echo "  (find failed)"
        
        echo ""
        echo "🐚 AVAILABLE SHELL COMMANDS:"
        docker run --platform "$platform" --rm "$full_image" sh -c 'which sh busybox ls cat find 2>/dev/null || echo "Basic commands not found"'
        
        echo ""
        echo "🔧 BUSYBOX ANALYSIS (if present):"
        docker run --platform "$platform" --rm "$full_image" sh -c 'if [ -x /bin/busybox ]; then /bin/busybox --list | head -10; else echo "No busybox found"; fi' 2>/dev/null || echo "  (busybox check failed)"
        
        echo ""
        echo "📦 ALTERNATIVE BINARY LOCATIONS:"
        for path in /usr/bin /bin /sbin /usr/sbin /usr/local/bin /app /opt/bin /system/bin; do
            echo -n "  $path: "
            docker run --platform "$platform" --rm "$full_image" sh -c "if [ -d $path ]; then ls $path 2>/dev/null | wc -l; else echo 'not found'; fi" 2>/dev/null || echo "error"
        done
        
        # Cleanup
        docker rm "$container_name" >/dev/null 2>&1 || true
        
    else
        echo "❌ Failed to create container for $full_image"
    fi
    
    echo ""
    echo "----------------------------------------"
done

echo ""
echo "🎯 EXPLORATION COMPLETE!"