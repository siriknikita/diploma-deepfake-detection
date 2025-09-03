#!/usr/bin/env bash

# This script updates the requirements.txt file with the current installed packages and their versions using uv command.
# Usage: ./update-requirements-txt-file.sh

uv pip compile pyproject.toml -o requirements.txt
