# legos
Ubuntu 24.04

Python 3.9.17


## Setup
1. Ensure python version is equal to 3.9.17 with `python3 --version`.
2. Create virtual environment `python3 -m venv .venv`.
3. Activate virtual environment `source .venv/bin/activate`.
4. Run `source env.sh` to get environment variables.



## Usage Info:
Set your run configuration in the configs yaml file.

Pass in the configuration name as an argument when running the script.

**Example**:
```
configuration_name:
  model: "Mistral7B"
  benchmark: Strategy"
  exemplars: "exemplars_subset_file_name.csv"
```


## Algorithm 2 - Insight Extraction
Usage: `python3 src/insight_extraction.py -c <config_name> -n <run_name>`

**Note**: The model and benchmark are set in `configs/insight_extraction.yaml`