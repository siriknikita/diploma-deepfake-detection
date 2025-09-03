#!/usr/bin/env bash

# This script runs black to check for formatting issues in the src/ directory.
# Usage: ./check-formatting.sh

echo "Checking code formatting with black..."
uv run black --check src/
