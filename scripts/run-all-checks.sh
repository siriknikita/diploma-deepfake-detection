#!/usr/bin/env bash

# This script runs all checks for the project, including formatting checks.
# Usage: ./run-all-checks.sh

bash -c './scripts/check-formatting.sh'
echo ''
bash -c './scripts/check-linting.sh'
echo ''
bash -c './scripts/check-static-types.sh'
