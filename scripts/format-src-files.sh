#!/usr/bin/env bash

# This script runs black to format the code in the src/ directory.
# Usage: ./format-src-files.sh

uv run black src/
