# src.utils

def load_config(config_file, config_name):
  import yaml
  with open(config_file, "r") as f:
    all_configs = yaml.safe_load(f)
    assert(all_configs.get(config_name, {}) != {})
  return all_configs.get(config_name, {})



def save_logs(benchmark: str, run_name: str, log_history: list, stats: dict, runtime: float):
  """
  Saves two versions of the logs:
    1. A full version (with staple prompt and all thoughts and actions).
    2. A cleaned version (with the staple prompt removed).
  """
  import os
  def _remove_template_prompt_from_log(log_text: str) -> str:
    """
    Removes all occurrences of the staple prompt block from the log text.
    Assumes each staple block starts with "You are QA system." and ends with "(END OF EXAMPLES)".
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
  full_log_path = os.path.join(benchmark_log_path, f"{run_name}_full_train.log")
  clean_log_path = os.path.join(benchmark_log_path, f"{run_name}_clean_train.log")
    
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
      f.write(str(stats))
      f.write(f"Runtime: {runtime} seconds")
    
  print(f"[TrainAgent] Full logs saved to {full_log_path}")
  print(f"[TrainAgent] Clean logs saved to {clean_log_path}")




def query(model: str, benchmark: str, exp_prompt: tuple):
  import src.prompts as prompts

  if benchmark == "StrategyQA":
    template_prompt = prompts.STRATEGYQA_PROMPT
  else:
    template_prompt = ""

  full_prompt = template_prompt + exp_prompt

  if model == "Mistral7B":
    from src.models.mistral7b import query_mistral7b
    output = query_mistral7b(full_prompt)

  return output



def compare_final_answer(task_text: str):
  import re
  # Extract the Answer field
  answer_match = re.search(r'^Answer:\s*(.+)$', task_text, re.MULTILINE)
  if not answer_match:
      raise ValueError("No Answer field found in the text.")
  answer = answer_match.group(1).strip()

  # Extract the Final Answer field; if there are multiple, take the last one.
  final_answer_matches = re.findall(r'^Final Answer:\s*(.+)$', task_text, re.MULTILINE)
  if not final_answer_matches:
      raise ValueError("No Final Answer field found in the text.")
  final_answer = final_answer_matches[-1].strip()

  # Compare the answers (case-insensitive)
  return "CORRECT" if answer.lower() == final_answer.lower() else "INCORRECT"