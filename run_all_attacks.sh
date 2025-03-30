#!/bin/bash

# Configuration
MODEL_PATH="/root/internlm2-chat"  # Adjust as needed
DEVICE="0"                        # GPU device ID
K_VALUE="7"                       # Value for the K parameter
CONFIG_FILE="behaviors_ours_config.json"
OUTPUT_PATH="./output_update_target"
NUM_BEHAVIORS=$(grep -o '\"behaviour\"' $CONFIG_FILE | wc -l)  # Count behaviors in the file

echo "Found $NUM_BEHAVIORS behaviors in $CONFIG_FILE"
echo "Starting attacks with K=$K_VALUE"

# Create a log directory
mkdir -p attack_logs

# Loop through all behavior IDs
for ID in $(seq 1 $NUM_BEHAVIORS); do
  echo "=============================================="
  echo "Starting attack for behavior ID $ID"
  echo "=============================================="
  
  # Run the attack script with current ID
  python attack_llm_core_best_update_our_target_v2.py \
    --model_path $MODEL_PATH \
    --device $DEVICE \
    --id $ID \
    --K $K_VALUE \
    --behaviors_config $CONFIG_FILE \
    --output_path $OUTPUT_PATH \
    2>&1 | tee attack_logs/attack_id${ID}.log
  
  # Check if the attack was successful
  if [ $? -eq 0 ]; then
    echo "Attack for ID $ID completed successfully"
  else
    echo "WARNING: Attack for ID $ID failed with error code $?"
  fi
  
  # Optional: Add a delay between runs to allow system to stabilize
  sleep 2
done

echo "All attacks completed!"
echo "Results saved to $OUTPUT_PATH/$K_VALUE/"

# Optional: Generate a summary report
echo "Generating summary report..."
echo "ID,Loss,Success" > $OUTPUT_PATH/$K_VALUE/summary.csv
for ID in $(seq 1 $NUM_BEHAVIORS); do
  LOSS_FILE="$OUTPUT_PATH/$K_VALUE/loss_logs/loss_log_${ID}.csv"
  if [ -f "$LOSS_FILE" ]; then
    FINAL_LOSS=$(tail -1 $LOSS_FILE | cut -d',' -f2)
    echo "$ID,$FINAL_LOSS,YES" >> $OUTPUT_PATH/$K_VALUE/summary.csv
  else
    echo "$ID,N/A,NO" >> $OUTPUT_PATH/$K_VALUE/summary.csv
  fi
done

echo "Summary saved to $OUTPUT_PATH/$K_VALUE/summary.csv" 
