#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
export BASE_DIR
export CONFIGS_DIR="$BASE_DIR/configs"

export DATA_DIR="$BASE_DIR/data"
export GSM8K_DIR="$DATA_DIR/gsm8k"
export STRATEGYQA_DIR="$DATA_DIR/strategyqa"