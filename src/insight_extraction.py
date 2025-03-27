import os
import yaml
import argparse

def main():
  # Reading config path
  config_path = os.getenv("CONFIGS_DIR") + "/insight_extraction.yaml"
  with open(config_path, "r") as f:
    data = yaml.safe_load(f)


if __name__ == "__main__":
  main()