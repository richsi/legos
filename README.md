# legos
Ubuntu 24.04

Python 3.9.17


## Setup

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

Usage: `python3 -m src.main -p train -c default -n test`

* `--phase, -p` - choose train, insight_extraction, or eval
* `--config, -c` - select which configuration in your yaml file you want to use
* `--run_name, -n` - name your run