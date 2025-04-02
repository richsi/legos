import os
import sys
import json
import pandas as pd

if __name__ == "__main__":

  args = sys.argv[1:]
  json_file = args[1]

  env_path = os.getenv(args[0].upper())
  json_path = os.path.join(env_path, json_file) 

  with open(json_path, "r") as f:
    data = json.load(f)

  
  df = pd.DataFrame(data)

  save_file_name = json_file[:-4] + "csv"
  save_file_path = os.path.join(env_path, save_file_name)

  df.to_csv(save_file_path, index=True)
  print(f"CSV file saved to {save_file_path}.")