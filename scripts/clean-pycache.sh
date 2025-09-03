#!/usr/bin/env bash

# This script removes all __pycache__ directories from the current directory and its subdirectories.
# Usage: ./clean-pycache.sh

find . -type d -name "__pycache__" -exec rm -r {} +
