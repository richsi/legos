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
  config_path = f"configs/{args.phase}.yaml"
  configs = utils.load_config(config_path, args.config)

  # Assigning variables 
  phase = args.phase
  run_name = args.run_name
  model = configs["model"]
  benchmark = configs["benchmark"]
  exemplars_file = os.getenv(benchmark.upper()) + "/" + configs["exemplars"]
  print(exemplars_file)
  # num_reflections = configs["num_reflections"]

  # Loading training data
  exemplars = pd.read_csv(exemplars_file)

  # Initializing model
  insight_agent = AGENT[phase](
    model=model,
    phase=phase,
    benchmark=benchmark,
    run_name=run_name,
    exemplars=exemplars,
    # num_reflections=num_reflections
  )

  insight_agent.run()

if __name__ == "__main__":
  main()