# src.utils

def load_config(config_file, model, dataset):
  import yaml
  config_name = "-".join([model, dataset])
  print(config_name)
  with open(config_file, "r") as f:
    all_configs = yaml.safe_load(f)
    assert(all_configs.get(config_name, {}) != {})
  return all_configs.get(config_name, {})


def save_logs(
  model: str,
  dataset: str, 
  run_name: str, 
  phase: str, 
  log_history: list, 
  stats: dict, 
  runtime: float, 
  total_token_sizes: list,
  eval_type: str=None,
  results_dict: dict=None
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
    Assumes each staple block starts with "(PROMPT START)." and ends with "(PROMPT END)".
    """
    start_marker = "(PROMPT START)"
    end_marker = "(PROMPT END)"
    
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
  dataset_log_path = os.path.join("logs", dataset)
  os.makedirs(dataset_log_path, exist_ok=True)
  phase_log_path = os.path.join(dataset_log_path, phase)
  os.makedirs(phase_log_path, exist_ok=True)

  full_log_path = os.path.join(phase_log_path, f"{run_name}_{model}_{phase}_full.log")
  clean_log_path = os.path.join(phase_log_path, f"{run_name}_{model}_{phase}_clean.log")
  if eval_type:
    eval_type_log_path = os.path.join(phase_log_path, eval_type)
    os.makedirs(eval_type_log_path, exist_ok=True)
    full_log_path = os.path.join(eval_type_log_path, f"{run_name}_{model}_{phase}_{eval_type}_full.log")
    clean_log_path = os.path.join(eval_type_log_path, f"{run_name}_{model}_{phase}_{eval_type}_clean.log")

  # Join all experience entries (each step) into one string
  full_log_text = "\n".join(log_history)
  # Process the log to remove the staple prompt blocks
  clean_log_text = _remove_template_prompt_from_log(full_log_text)
    
  with open(full_log_path, "w") as f:
      f.write(full_log_text + "\n")
      f.write(str(stats) + "\n")
      f.write(f"Runtime: {runtime} seconds\n")
      f.write(f"Total token size: {sum(total_token_sizes)}\n")
      f.write(f"Average token size: {sum(total_token_sizes) // len(total_token_sizes)}\n")
  with open(clean_log_path, "w") as f:
      f.write(clean_log_text + "\n")
      f.write(str(stats) + "\n")
      f.write(f"Runtime: {runtime} seconds\n")
      f.write(f"Total token size: {sum(total_token_sizes)}\n")
      f.write(f"Average token size: {sum(total_token_sizes) // len(total_token_sizes)}\n")
    
  print(f"[TrainAgent] Full logs saved to {full_log_path}")
  print(f"[TrainAgent] Clean logs saved to {clean_log_path}")

  # CSV Logging
  
  if phase == "eval":
    import pandas as pd
    csv_logfile = os.path.join(os.getenv("LOGS"), "results.csv")
    pd.DataFrame(results_dict).to_csv(csv_logfile, mode='a', index=False, header=False)
    print(f"CSV file has been saved to {csv_logfile}")



def format_prompt(phase: str, dataset: str, **kwargs):
  from src.prompts import PROMPTS

  if phase == "insight_extraction":
    base_prompt = PROMPTS["_".join([dataset, phase])]
    full_prompt = base_prompt.format(kwargs["exemplars"])
  elif phase == "eval":
    base_prompt = PROMPTS["_".join([dataset, phase, kwargs["eval_type"]])]
    if kwargs["eval_type"] == "insight":
      full_prompt = base_prompt.format(kwargs["insights"], kwargs["test_data"])
    elif kwargs["eval_type"] == "exemplar":
      full_prompt = base_prompt.format(kwargs["exemplars"], kwargs["test_data"])
    elif kwargs["eval_type"] == "insight_exemplar":
      full_prompt = base_prompt.format(kwargs["exemplars"], kwargs["insights"],kwargs["test_data"])

  return full_prompt


def query(model: str, prompt: str):
  from src.models import QUERY
  return QUERY[model](prompt)


def get_insights(model: str, dataset: str, run_name: str):
  import os
  import re
  file_name = f"{run_name}_{model}_insight_extraction_clean.log"
  insights_path = os.path.join("logs", dataset, "insight_extraction", file_name)
  # Pattern to match lines that start with "RULE" or a digit (e.g., "1." or "2.")
  rule_pattern = re.compile(r'^(RULE|\d+\.)', re.IGNORECASE)
    
  with open(insights_path, "r") as f:
    insights = []
    for line in f:
      # Check if the line starts with "RULE" or a digit.
      if rule_pattern.match(line.lstrip()):
        insights.append(line.strip())
  return "\n".join(insights)


def count_tokens(text, model="gpt-3.5-turbo"):
  import tiktoken
  encoding = tiktoken.encoding_for_model(model)
  return len(encoding.encode(text))


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
