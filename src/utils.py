# src.utils

def load_config(config_file, config_name):
  import yaml
  with open(config_file, "r") as f:
    all_configs = yaml.safe_load(f)
    assert(all_configs.get(config_name, {}) != {})
  return all_configs.get(config_name, {})




def save_logs(
  model: str,
  benchmark: str, 
  run_name: str, 
  phase: str, 
  log_history: list, 
  stats: dict, 
  runtime: float, 
  results_dict: dict
):
  """
  Saves two versions of the logs:
    1. A full version (with staple prompt and all thoughts and actions).
    2. A cleaned version (with the staple prompt removed).
  """
  import os
  def _remove_template_prompt_from_log(log_text: str) -> str:
    """
    Removes all occurrences of the staple prompt block from the log text.
    Assumes each staple block starts with "(TEMPLATE START)." and ends with "(TEMPLATE END)".
    """
    start_marker = "(TEMPLATE START)"
    end_marker = "(TEMPLATE END)"
    
    # Continue removing until no complete staple block remains.
    while start_marker in log_text and end_marker in log_text:
        start_index = log_text.find(start_marker)
        end_index = log_text.find(end_marker, start_index)
        if start_index == -1 or end_index == -1:
            break
        # Remove from start_marker through the end_marker (include end_marker)
        log_text = log_text[:start_index] + log_text[end_index + len(end_marker):]
    return log_text

  os.makedirs("logs", exist_ok=True)
  benchmark_log_path = os.path.join("logs", benchmark)
  os.makedirs(benchmark_log_path, exist_ok=True)
  phase_log_path = os.path.join(benchmark_log_path, phase)
  os.makedirs(phase_log_path, exist_ok=True)
  full_log_path = os.path.join(phase_log_path, f"{run_name}_{model}_{phase}_full.log")
  clean_log_path = os.path.join(phase_log_path, f"{run_name}_{model}_{phase}_clean.log")
    
  # Join all experience entries (each step) into one string
  full_log_text = "\n".join(log_history)
  # Process the log to remove the staple prompt blocks
  clean_log_text = _remove_template_prompt_from_log(full_log_text)
    
  with open(full_log_path, "w") as f:
      f.write(full_log_text + "\n")
      f.write(str(stats) + "\n")
      f.write(f"Runtime: {runtime} seconds")
  with open(clean_log_path, "w") as f:
      f.write(clean_log_text + "\n")
      f.write(str(stats) + "\n")
      f.write(f"Runtime: {runtime} seconds")
    
  print(f"[TrainAgent] Full logs saved to {full_log_path}")
  print(f"[TrainAgent] Clean logs saved to {clean_log_path}")

  # CSV Logging
  
  # import pandas as pd
  # log_dir = os.path.join(os.getenv("LOGS"), benchmark, phase)
  # csv_logfile = f"{log_dir}/{run_name}_{model}_{phase}.csv"
  # pd.DataFrame(results_dict).to_csv(csv_logfile, index=False)
  # print(f"CSV file has been saved to {csv_logfile}")

def format_prompt(phase: str, benchmark: str, **kwargs):
  from src.prompts import PROMPTS
  base_prompt = PROMPTS[benchmark + "_" + phase]

  if phase == "train":
    full_prompt = base_prompt
  elif phase == "insight_extraction":
    full_prompt = base_prompt.format(kwargs["exemplars"])
  elif phase == "eval":
    full_prompt = base_prompt.format(kwargs["exemplars"], kwargs["insights"], kwargs["test_data"])
  return full_prompt


def query(model: str, prompt: str):
  from src.models import QUERY
  return QUERY[model](prompt)



def get_insights(model: str, benchmark: str, run_name: str):
  import os
  import re
  file_name = "_".join([run_name, model, "insight_extraction", "clean.log"])
  insights_path = os.path.join("logs", benchmark, "insight_extraction", file_name)
  
  with open(insights_path, "r") as f:
     insights = []
     for line in f:
        if re.match(r'^\d+\.\s', line):
          insights.append(line.strip())
  return "\n".join(insights)


def self_consistency():
  # TODO: implement
  """
  def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        ans = ""
        if len(tmp.split("Final Answer:"))>0:
            ans = tmp.split("Final Answer:")[-1]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "******")
            try:
                float(ans)
                ans = round(float(ans))
                ans_list.append(ans)
            except: pass
        # ans_list.append(ans)

    # print(ans_list)
    d = {}
    for i in ans_list:
        if i=="":
            continue
        if int(i) in d:
            d[int(i)] += 1
        else:
            d[int(i)] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n
  """
  pass
