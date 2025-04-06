#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
export BASE_DIR

export DATA="$BASE_DIR/data"
export STRATEGYQA="$DATA/strategyqa"
export GSM8K="$DATA/gsm8k"
export TABMWP="$DATA/tabmwp"
export AQUARAT="$DATA/aquarat"
export FINQA="$DATA/finqa"

export LOGS="$BASE_DIR/logs"