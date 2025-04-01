#!/bin/bash
# Define variables for better maintainability
BASE_PATH="./results/1_jbk/7/log"
MODEL_PATH="/root/internlm2-chat"
JUDGE_MODEL_PATH="/root/autodl-tmp/beaver-dam-7b"

# Correct bash sequence syntax
for idx in {1..30}
do
    echo "Processing file index: $idx"
    python evaluate_intern_llm.py \
        --file_idx "$idx" \
        --base_path "$BASE_PATH" \
        --model_path "$MODEL_PATH" \
        --judge_model_path "$JUDGE_MODEL_PATH"
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error processing index $idx"
    fi
done

echo "Evaluation complete!"