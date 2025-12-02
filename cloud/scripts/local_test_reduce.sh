#!/bin/bash
# Local test script for reduce phase container
#
# Usage: ./cloud/scripts/local_test_reduce.sh [INPUT_DIR] [OUTPUT_FILE]
#
# Examples:
#   ./cloud/scripts/local_test_reduce.sh output/map_test output/reduce_test/themes.json
#   ./cloud/scripts/local_test_reduce.sh  # Uses defaults

set -e

# Defaults
INPUT_DIR="${1:-output/map_test}"
OUTPUT_FILE="${2:-output/reduce_test/themes.json}"
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")

echo "=== Reduce Phase Local Test ==="
echo "Input Dir: $INPUT_DIR"
echo "Output File: $OUTPUT_FILE"
echo ""

# Check input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    echo "Run map phase first: ./cloud/scripts/local_test_map.sh"
    exit 1
fi

# Count input files
JSON_COUNT=$(ls -1 "$INPUT_DIR"/*_topics.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "ERROR: No *_topics.json files found in $INPUT_DIR"
    exit 1
fi
echo "Found $JSON_COUNT firm topic files"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the entrypoint directly (without Docker)
echo "Running reduce phase..."
LOCAL_MODE=true \
LOCAL_INPUT="$INPUT_DIR" \
LOCAL_OUTPUT="$OUTPUT_FILE" \
MIN_FIRMS=1 \
LOG_LEVEL=INFO \
python -m cloud.containers.reduce.entrypoint

echo ""
echo "=== Output ==="
if [ -f "$OUTPUT_FILE" ]; then
    echo "Successfully created: $OUTPUT_FILE"
    echo ""
    echo "Summary:"
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    themes = json.load(f)
print(f'  Themes discovered: {len(themes)}')
if themes:
    total_topics = sum(t['n_topics'] for t in themes)
    total_firms = len(set(t['firm_id'] for theme in themes for t in theme['topics']))
    print(f'  Total topics in themes: {total_topics}')
    print(f'  Unique firms: {total_firms}')
    print('')
    print('  Top themes:')
    for i, theme in enumerate(themes[:5]):
        name = theme['name'][:50] if len(theme['name']) > 50 else theme['name']
        print(f'    {theme[\"theme_id\"]}: {name} ({theme[\"n_topics\"]} topics, {theme[\"n_firms\"]} firms)')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi
