import os
import argparse
import pandas as pd
import src.utils as utils
import src.models as models
from src.agents import AGENT

def main():
  # Getting arguments / configs
  parser = argparse.ArgumentParser()
  parser.add_argument("--phase", "-p", type=str, default="eval", 
                      help="insight_extraction or eval")
  parser.add_argument("--eval_type", "-e", type=str, required=False, 
                      help="insight, exemplar, insight_exemplar")
  parser.add_argument("--model", "-m", type=str, required=True, 
                      help="mistral7b, llama3b, llama1b")
  parser.add_argument("--dataset", "-d", type=str, required=True, 
                      help="strategyqa, gsm8k, tabmwp, aquarat, finqa")
  parser.add_argument("--run_name", "-n", type=str, required=True, 
                      help="Name your run")
  parser.add_argument("--sc", "-sc", type=bool, required=False, 
                      help="Name whether self consistency is needed")

  args = parser.parse_args()

  # Loading yaml file
  configs = utils.load_config("configs.yaml", args.model, args.dataset)

  kwargs = configs 
  kwargs["phase"] = args.phase
  kwargs["model"] = args.model
  kwargs["dataset"] = args.dataset
  kwargs["run_name"] = args.run_name
  if args.sc is None:
    kwargs["sc"] = False
  else:
    kwargs["sc"] = args.sc

  if args.phase == "eval":
    assert(args.eval_type != None)
    kwargs["eval_type"] = args.eval_type
  else:
    assert(args.eval_type == None)

  print(kwargs)
  # Initializing model
  agent = AGENT[args.phase](**kwargs)

  agent.run()

if __name__ == "__main__":
  main()