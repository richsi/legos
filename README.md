# legos

Exploration of improving benchmark performance with reduced computational overhead through coordination of an ensemble of smaller LLM models.

## Overview


## Components

### Core 


### Agents


### Models


### Prompts


## Building the Project

### Prerequisites
* Ubuntu 24.04
* Python 3.9.17

### Setup

#### Python venv
1. Ensure python version is equal to 3.9.17 with `python3 --version`.
2. Create virtual environment `python3 -m venv .venv`.
3. Activate virtual environment `source .venv/bin/activate`.
4. Run `source env.sh` to set environment variables.
5. Set up huggingface token with `huggingface-cli login`.
6. `pip install -r requirements.txt`

## Usage Info:
Set your run configuration in the configs yaml file.

Pass in the configuration name as an argument when running the script.

**Example Configuration**:
```
configuration_name:
  model: "Mistral7B"
  benchmark: Strategy"
  exemplars: "exemplars_subset_file_name.csv"
```

Insight Extraction: `python3 -m src.main -p insight_extraction -c <config_name> -n <run_name>`

Evaluation: `python3 -m src.main -p eval -c <config_name> -n <run_name>`

* `--phase, -p` - choose train, insight_extraction, or eval
* `--config, -c` - select which configuration in your yaml file you want to use
* `--run_name, -n` - name your run, will also be your log file name


Keep the run_name constant across training, insight extraction, and evaluation. The logging script automatically adds the model name and phase to the end of the run_name.

[ ] StrategyQA
[ ] GSM8k
[ ] AquaRat
[ ] TabMWP
[ ] FinaQA
[ ] Use base.yaml to store training exemplar path