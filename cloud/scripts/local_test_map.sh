#!/bin/bash
# Local test script for map phase container
#
# Usage: ./cloud/scripts/local_test_map.sh [FIRM_ID] [CSV_PATH]
#
# Examples:
#   ./cloud/scripts/local_test_map.sh 1001 transcripts.csv
#   ./cloud/scripts/local_test_map.sh  # Uses defaults

set -e

# Defaults
FIRM_ID="${1:-1001}"
CSV_PATH="${2:-transcripts_2023-01-01_to_2023-03-31_enriched.csv}"
OUTPUT_DIR="output/map_test"

echo "=== Map Phase Local Test ==="
echo "Firm ID: $FIRM_ID"
echo "CSV Path: $CSV_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the entrypoint directly (without Docker)
echo "Running map phase..."
LOCAL_MODE=true \
FIRM_ID="$FIRM_ID" \
LOCAL_INPUT="$CSV_PATH" \
LOCAL_OUTPUT="$OUTPUT_DIR" \
LOG_LEVEL=INFO \
python -m cloud.containers.map.entrypoint

echo ""
echo "=== Output ==="
if [ -f "$OUTPUT_DIR/${FIRM_ID}_topics.json" ]; then
    echo "Successfully created: $OUTPUT_DIR/${FIRM_ID}_topics.json"
    echo ""
    echo "Summary:"
    python -c "
import json
with open('$OUTPUT_DIR/${FIRM_ID}_topics.json') as f:
    data = json.load(f)
print(f'  Firm: {data[\"firm_name\"]} ({data[\"firm_id\"]})')
print(f'  Topics: {data[\"n_topics\"]}')
print(f'  Sentences processed: {data[\"metadata\"][\"n_sentences_processed\"]}')
print(f'  Outliers: {len(data[\"outlier_sentence_ids\"])}')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi
