#!/usr/bin/env bash

# This script runs ruff to check for linting issues in the src/ directory.
# Usage: ./check-linting.sh

uv run ruff check src/
