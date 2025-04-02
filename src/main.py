import os
import argparse
import pandas as pd
import src.utils as utils
import src.models as models
from src.agents import AGENT

def main():
  # Getting arguments / configs
  parser = argparse.ArgumentParser()
  parser.add_argument("--phase", "-p", type=str, default="train", 
                      help="train, insight_extraction, or eval?")
  parser.add_argument("--config", "-c", type=str, required=True, 
                      help="Which configuration to use from insight_extractions.yaml")
  parser.add_argument("--run_name", "-n", type=str, required=True, 
                      help="Name your run")
  args = parser.parse_args()

  # Loading yaml file
  configs = utils.load_config("configs.yaml", args.config)

  kwargs = configs 
  kwargs["phase"] = args.phase
  kwargs["run_name"] = args.run_name

  # Initializing model
  agent = AGENT[args.phase](**kwargs)

  agent.run()

if __name__ == "__main__":
  main()