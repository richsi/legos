
def load_config(config_file, config_name):
  import yaml
  with open(config_file, "r") as f:
    all_configs = yaml.safe_load(f)
    assert(all_configs.get(config_name, {}) != {})
  return all_configs.get(config_name, {})




def save_logs(benchmark: str, run_name: str, log_history: list):
  """
  Saves two versions of the logs:
    1. A full version (with staple prompt and all thoughts and actions).
    2. A cleaned version (with the staple prompt removed).
  """
  import os
  def _remove_staple_prompt_from_log(log_text: str) -> str:
    """
    Removes all occurrences of the staple prompt block from the log text.
    Assumes each staple block starts with "You are QA system." and ends with "(END OF EXAMPLES)".
    """
    start_marker = "You are QA system."
    end_marker = "(END OF EXAMPLES)"
    
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
  full_log_path = os.path.join(benchmark_log_path, f"{run_name}_full_train.log")
  clean_log_path = os.path.join(benchmark_log_path, f"{run_name}_clean_train.log")
    
  # Join all experience entries (each step) into one string
  full_log_text = "\n".join(log_history)
  # Process the log to remove the staple prompt blocks
  clean_log_text = _remove_staple_prompt_from_log(full_log_text)
    
  with open(full_log_path, "w") as f:
      f.write(full_log_text)
  with open(clean_log_path, "w") as f:
      f.write(clean_log_text)
    
  print(f"[TrainAgent] Full logs saved to {full_log_path}")
  print(f"[TrainAgent] Clean logs saved to {clean_log_path}")
