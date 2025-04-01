#!/bin/bash

# Check if config file path is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_json_path>"
    exit 1
fi

CONFIG_FILE=$1
# Extract the prefix before "_config.json"
PREFIX=$(echo "$CONFIG_FILE" | sed 's/_config\.json$//')
OUTPUT_DIR="./challenge_output/${PREFIX}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Read the JSON file and process each item
python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
    for item in config:
        id = item['id']
        print(f'Processing item {id}')
        print('python attack_llm_core_best_update_our_target_v4.py',
              f'--id {id}',
              f'--behaviors_config {CONFIG_FILE}',
              f'--output_path {OUTPUT_DIR}')"
" | while read cmd; do
    echo "Executing: $cmd"
    eval "$cmd"
done

