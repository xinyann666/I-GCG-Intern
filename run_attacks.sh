#!/bin/bash

# Trap Ctrl+C and terminate any running child processes
trap "echo 'Script interrupted, stopping all processes...'; pkill -f 'attack_llm_core_best_update_our_target_v5.py'; exit 1" INT

# Loop through config files i = 1 to 6
for i in {1..6}; do
    CONFIG_FILE="./data/${i}_jbk_config.json"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file $CONFIG_FILE does not exist. Skipping."
        continue
    fi

    # Extract the prefix from the config file name (e.g., "1_jbk_config.json" -> "1_jbk")
    PREFIX=$(basename "$CONFIG_FILE" | sed 's/_config\.json$//')
    OUTPUT_DIR="./results/${PREFIX}"
    mkdir -p "$OUTPUT_DIR"

    echo "Processing config file: $CONFIG_FILE"
    echo "Output directory: $OUTPUT_DIR"

    # Process each item in the config file using an embedded Python script.
    python3 <<EOF
import json
import os
import sys

config_file = "$CONFIG_FILE"
output_dir = "$OUTPUT_DIR"

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error reading config file: {e}")
    sys.exit(1)

if not isinstance(config, list):
    print("Config file must be a JSON list of items.")
    sys.exit(1)

for item in config:
    id = item.get('id')
    if id is None:
        print("Skipping an item without an 'id'.")
        continue
    print(f"Processing item {id}")
    command = f"python attack_llm_core_best_update_our_target_v5.py --id {id} --behaviors_config {config_file} --output_path {output_dir} --target_len_max 60"
    print("Executing:", command)
    ret = os.system(command)
    if ret != 0:
        print(f"Command failed with exit code {ret}")
EOF

done

echo "All config files processed. Done!"