#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
export BASE_DIR
export CONFIGS_DIR="$BASE_DIR/configs"

export DATA="$BASE_DIR/data"
export GSM8K="$DATA/gsm8k"
export STRATEGYQA="$DATA/strategyqa"
export LOGS="$BASE_DIR/logs"