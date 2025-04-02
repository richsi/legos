# legos

Exploration of improving benchmark performance with reduced computational overhead through coordination of an ensemble of smaller LLM models.

## TODO
- [x] Only Insight (based on exemplars)
- [x] Training time
- [ ] Only exemplars (EXPLORA baseline)
- [ ] Exemplar + Insight 
- [ ] Insight + Exemplar 
- [ ] Reduced Exemplar list (based on cosine similarity) + Exemplar
- [ ] Repeat using FP=16 and FP=32
- [ ] Additional measurements
- [ ] Token / context size


## Components
### Agents
* `insight.py`: Insight extraction agent that generates insights based on configuration benchmark and model
* `eval.py`: Evaluation agent that processes generated insights to answer test questions and record results

### Models
* `mistral7b.py`: Mistral-7B-Instruct-v0.1
* `llama3b`: Llama-3.2-3b-Instruct
* `llama1b`: Llama-3.2-1b-Instruct

### Prompts
* `strategyqa_prompts`: contains insight extraction and evaluation prompts for StrategyQA dataset
* `gsm8k_prompts`: contains insight extraction and evaluation prompts for GSM8K dataset
* `tabmwp_prompts`: contains insight extraction and evaluation prompts for TabMWP dataset

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

**Example** `config.py`:
```
configuration_name:
  model: "Mistral7B"
  benchmark: StrategyQA"
  train: "exemplars_subset_file_name.csv"
  eval: "test_file_name.csv"
```

Insight Extraction: `python3 -m src.main -p insight_extraction -c <config_name> -n <run_name>`

Evaluation: `python3 -m src.main -p eval -c <config_name> -n <run_name>`

* `--phase, -p` - choose train, insight_extraction, or eval
* `--config, -c` - select which configuration in your yaml file you want to use
* `--run_name, -n` - name your run, will also be your log file name