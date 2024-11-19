#!/bin/bash

# Default values
CSV_DIR="../data/llm_judge"
TEMPLATE_PATH="../prompt/llm_pairwise_judge_v1.txt"
MODELS=()
CONFIG_LABELS=("ab1" "ab2" "ab3" "ba1" "ba2" "ba3")
CONFIG_COLUMNS=(
  "corrected_input_text response_a response_b"
  "corrected_input_text response_a response_b"
  "corrected_input_text response_a response_b"
  "corrected_input_text response_b response_a"
  "corrected_input_text response_b response_a"
  "corrected_input_text response_b response_a"
)

# Function to display help
usage() {
  echo "Usage: $0 -m model1,model2,... -d CSV_DIR -t TEMPLATE_PATH"
  exit 1
}

# Parse named arguments
while getopts "m:d:t:h" opt; do
  case $opt in
  m) IFS=',' read -r -a MODELS <<<"$OPTARG" ;;
  d) CSV_DIR="$OPTARG" ;;
  t) TEMPLATE_PATH="$OPTARG" ;;
  h) usage ;;
  *) usage ;;
  esac
done

# Check if MODELS array is empty
if [ ${#MODELS[@]} -eq 0 ]; then
  echo "Error: At least one model must be specified."
  usage
fi

# Loop through each model
for MODEL_NAME in "${MODELS[@]}"; do

  # Loop through each configuration
  for i in "${!CONFIG_LABELS[@]}"; do
    LABEL="${CONFIG_LABELS[$i]}"
    TEXT_COLUMNS="${CONFIG_COLUMNS[$i]}"

    # Find the most recently modified CSV file
    INPUT_CSV=$(ls -t "$CSV_DIR"/*.csv | head -1)

    # Run the llm_annotation.py script with the current configuration
    python ../src/llm_annotation.py \
      --input-path "$INPUT_CSV" \
      --output-path "$CSV_DIR" \
      --prompt-template "$TEMPLATE_PATH" \
      --text-columns $TEXT_COLUMNS \
      --annotator-models "$MODEL_NAME" \
      --log-steps 10

    # Find the most recently modified CSV file again (after running the script)
    OUTPUT_CSV=$(ls -t "$CSV_DIR"/*.csv | head -1)

    # Rename columns in the CSV file
    python - <<END
import pandas as pd

df = pd.read_csv("$OUTPUT_CSV")

# Rename columns, appending the model name and configuration label
df.rename(columns={
    "pred_${MODEL_NAME}": "pred_${MODEL_NAME}_${LABEL}"
}, inplace=True)

df.to_csv("$OUTPUT_CSV", index=False)
END

  done

done
