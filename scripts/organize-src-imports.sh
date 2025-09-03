#!/usr/bin/env bash

# This script runs ruff to organize imports in the src/ directory.
# Usage: ./organize-src-imports.sh

uv run ruff --fix src/
