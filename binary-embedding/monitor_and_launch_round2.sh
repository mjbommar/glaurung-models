#!/usr/bin/bash

# Monitor Round 1 experiments and auto-launch Round 2 when complete

echo "========================================="
echo "EXPERIMENT MONITOR"
echo "Waiting for Round 1 to complete..."
echo "Checking every 10 minutes"
echo "========================================="

ROUND1_DIR=$(ls -td experiments_* 2>/dev/null | head -1)

if [ -z "$ROUND1_DIR" ]; then
    echo "ERROR: No experiments directory found"
    exit 1
fi

echo "Monitoring: $ROUND1_DIR"
echo ""

# Function to check if all 8 experiments are complete
check_completion() {
    local dir=$1
    local completed=0

    for i in $(seq -w 1 8); do
        exp_dir="${dir}/exp${i}_*"

        # Check if final_model directory exists (indicates completion)
        if ls ${exp_dir}/final_model 2>/dev/null | grep -q .; then
            ((completed++))
        fi
    done

    echo "$completed"
}

# Monitor loop
while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    completed=$(check_completion "$ROUND1_DIR")

    echo "[$timestamp] Round 1 progress: $completed/8 experiments complete"

    if [ "$completed" -eq 8 ]; then
        echo ""
        echo "========================================="
        echo "✓ Round 1 COMPLETE!"
        echo "Launching Round 2 experiments..."
        echo "========================================="
        echo ""

        # Launch Round 2
        export CUDA_VISIBLE_DEVICES=0
        bash run_experiments_round2.sh

        echo ""
        echo "========================================="
        echo "✓ Round 2 COMPLETE!"
        echo "========================================="

        exit 0
    fi

    # Sleep for 10 minutes
    echo "  Sleeping for 10 minutes..."
    sleep 600
    echo ""
done
