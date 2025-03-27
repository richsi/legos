#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
export BASE_DIR
export CONFIGS_DIR="$BASE_DIR/configs"