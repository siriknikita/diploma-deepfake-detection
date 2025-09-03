#!/usr/bin/env bash

# This script runs mypy to check for static type issues in the src/ directory.
# Usage: ./check-static-types.sh

echo "Checking static types with mypy..."
uv run mypy src/
