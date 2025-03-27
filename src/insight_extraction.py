import os
import yaml
import argparse
import pandas as pd

def load_config(config_file, config_name):
  with open(config_file, "r") as f:
    all_configs = yaml.safe_load(f)
    assert(all_configs.get(config_name, {}) != {})
  return all_configs.get(config_name, {})


def main():
  # Getting arguments / configs
  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", "-p", type=str, default="configs/insight_extraction.yaml", 
                      help="Path to insight_extractions.yaml")
  parser.add_argument("--config", "-c", type=str, required=True, 
                      help="Which configuration to use from insight_extractions.yaml")
  parser.add_argument("--run_name", "-n", type=str, required=True, 
                      help="Name your run")

  args = parser.parse_args()
  configs = load_config(args.config_path, args.config)

  # Assigning variables 
  model = configs["model"]
  benchmark = configs["benchmark"]
  exemplars_file = os.getenv("STRATEGYQA_DIR") + "/" + configs["exemplars"]

  # Loading training data
  exemplars = pd.read_csv(exemplars_file)


if __name__ == "__main__":
  main()