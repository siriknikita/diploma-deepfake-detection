#!/usr/bin/env bash

# This script runs ruff to check for linting issues in the src/ directory.
# Usage: ./check-linting.sh

echo "Checking code linting with ruff..."
uv run ruff check src/
