#!/usr/bin/env bash

# This script runs mypy to check for static type issues in the src/ directory.
# Usage: ./check-static-types.sh

uv run mypy src/
