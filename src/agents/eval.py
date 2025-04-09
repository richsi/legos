import ast
import os
import re
import time
import pandas as pd
from src.agents.base import BaseAgent
import src.utils as utils

class EvalAgent(BaseAgent):
  def __init__(
    self,
    **kwargs
  ):
    # Default variables
    self.model = kwargs["model"]
    self.phase = kwargs["phase"]
    self.dataset = kwargs["dataset"]
    self.run_name = kwargs["run_name"]
    self.eval_type = kwargs["eval_type"]
    self.train_exemplars = pd.read_csv(os.path.join(os.getenv(kwargs["dataset"].upper()), kwargs["train"])) # pd.DataFrame type

    self.num_tasks = 0
    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.runtime = 0
    self.stats = {"CORRECT": 0, "INCORRECT": 0, "FAILED":0}

    self.eval_file = kwargs["eval"]
    self.eval_df = None
    self.all_test_exemplars = []

    self.total_token_sizes = []
    self.error_log = []


  def run(self, reset: bool=True):
    """
    Starts new run
    """
    # Dict mapping to correct exemplar getter function
    try:
      exemplar_getter = {
        "strategyqa": self.get_strategyqa_exemplars,
        "gsm8k": self.get_gsm8k_exemplars,
        "tabmwp": self.get_tabmwp_exemplars,
        "aquarat": self.get_aquarat_exemplars,
        "finqa": self.get_finqa_exemplars,
      }.get(self.dataset)
      all_train_exemplars = [exemplar_getter(row) for _, row in self.train_exemplars.iterrows()]
      train_data = "\n\n".join(all_train_exemplars)
    except Exception as e:
      error_msg = f"Error getting train exemplars for {self.dataset}: {e}"
      self.error_log.append(error_msg)
      utils.log_errors(self.error_log, self.dataset, self.run_name, self.model, self.phase)
      return

    # Getting the eval file path
    eval_path = os.path.join(os.getenv(self.dataset.upper()), self.eval_file)
    self.eval_df = pd.read_csv(eval_path)

    # Formatting the test exemplars to pass into prompt
    try:
      test_exemplar_getter = {
        "strategyqa": self.get_strategyqa_test_exemplars,
        "gsm8k": self.get_gsm8k_test_exemplars,
        "tabmwp": self.get_tabmwp_test_exemplars,
        "aquarat": self.get_aquarat_test_exemplars,
        "finqa": self.get_finqa_test_exemplars
      }.get(self.dataset)
      self.all_test_exemplars = [test_exemplar_getter(row) for _, row in self.eval_df.iterrows()]
      self.num_tasks = len(self.all_test_exemplars) 
    except Exception as e:
      error_msg = f"Error getting test exemplars for {self.dataset}: {e}"
      self.error_log.append(error_msg)
      utils.log_errors(self.error_log, self.dataset, self.run_name, self.model, self.phase)
      return

    # Getting insights from same run_name from logs/insight_extraction
    try:
      insights = utils.get_insights(self.model, self.dataset, self.run_name)
    except Exception as e:
      error_msg = f"Error extracting insights from {self.run_name}_{self.model}_insight_clean.log for dataset {self.dataset}:\n{e}"
      self.error_log.append(error_msg)
      insights = ""
    # print(f"INSIGHTS:\n{insights}")

    kwargs = dict(
      exemplars=train_data,
      insights=insights,
      eval_type=self.eval_type,
    )

    start_time = time.time()
    while not self.done():
      self.step(**kwargs)
    end_time = time.time()
    self.runtime = end_time - start_time

    try:
      EM = self.stats["CORRECT"] / (self.stats["CORRECT"] + self.stats["INCORRECT"] + self.stats["FAILED"]) * 100 
    except Exception as e:
      error_msg = f"Unable to compute EM: {e}"
      self.error_log.append(error_msg)
      EM = -1 # error

    try:
      results_dict = {
        "matches": [self.stats["CORRECT"]],
        "mismatches": [self.stats["INCORRECT"] + self.stats["FAILED"]],
        "EM": [EM],
        "train_data_len": [len(all_train_exemplars)],
        "test_data_len": [len(self.all_test_exemplars)],
        "model": [self.model],
        "dataset": [self.dataset],
        "run_name": [self.run_name],
        "avg_token_len": [sum(self.total_token_sizes) / len(self.total_token_sizes)],
        "eval_type": [self.eval_type],
      }

      utils.save_logs(
        self.model,
        self.dataset, 
        self.run_name, 
        self.phase,
        self.log_history, 
        self.stats, 
        self.runtime,
        self.total_token_sizes,
        self.eval_type,
        results_dict
      )  
    except Exception as e:
      error_msg = f"Error saving logs for {self.dataset}/{self.run_name}_{self.model}_eval_{self.eval_type}: {e}"
      self.error_log.append(error_msg)
    
    if self.error_log:
      utils.log_errors(self.error_log, self.dataset, self.run_name, self.model, self.phase)

  def step(self, **kwargs):
    """
    Log:
      - The static task description prompt
      - Exemplar details (from the CSV)
      - Generated insights from the exemplars
      - Final evaluation answers
    """

    kwargs["test_data"] = self.all_test_exemplars[self.task_idx]
    # Formatting prompt for LLM
    prompt = utils.format_prompt(self.phase, self.dataset, **kwargs) 

    try:
      llm_output = utils.query(self.model, prompt)
      self.total_token_sizes.append(utils.count_tokens(llm_output))
    except Exception as e:
      error_msg = f"Error querying {self.model} for task {self.task_idx}: {e}"
      self.error_log.append(error_msg)
      llm_output = ""
      self.total_token_sizes.append(-1)

    # Recording the stats
    try:
      self.record_stats(llm_output, len(prompt))
    except Exception as e:
      error_msg = f"Error recording stats for task {self.task_idx}: {e}"
      self.error_log.append(error_msg)

    # Combine all elements into an experience log entry
    experience_log = (
        f"{self.model} Task {self.task_idx}:\n{llm_output}\n\nToken count: {self.total_token_sizes[self.task_idx]}\n"
        "-------------------------------------"
    )
    # Save and print the experience log
    self.log_history.append(experience_log)
    # increment task index
    self.task_idx += 1

  def done(self):
    # return self.task_idx >= self.num_tasks
    return self.task_idx >= 2

  def get_strategyqa_exemplars(self, exemplar):
    string = "Facts: "
    facts_list = ast.literal_eval(exemplar["facts"])
    for fact in facts_list:
      string += "\n" + fact
    string += "\n\nQuestion: " + exemplar["question"]
    string += "\nAnswer:\n"
    decomp_list = ast.literal_eval(exemplar["decomposition"])
    for idx, question in enumerate(decomp_list):
      string += "Sub-question {}: {}\n".format(idx+1, question)
    string += "The answer is: " + exemplar["answer"]
    return string
  
  def get_strategyqa_test_exemplars(self, exemplar):
    string = "Facts: "
    facts_list = ast.literal_eval(exemplar["facts"])
    for fact in facts_list:
      string += "\n" + fact
    string += "\n\nQuestion: " + exemplar["question"]
    return string

  def get_gsm8k_exemplars(self, exemplar):
    answer = re.sub("<<.*?>>", "", exemplar["answer"])
    answer = answer.replace("####", "Final Answer:")
    string = "Question: " + exemplar["question"] + "\n" + answer + "\n\n"
    return string

  def get_gsm8k_test_exemplars(self, exemplar):
    string = "Question: " + exemplar["question"] + "\n"
    return string

  def get_tabmwp_exemplars(self, exemplar):
    choices_str = "Please select from the following options: " + exemplar["choices"] \
                  if type(exemplar["choices"]) == str else ""
    return "Table: \n{}\nQuestion: {}\n{}\nAnswer: {}\nThe answer is: {}".format(
      exemplar["table"],
      exemplar["question"],
      choices_str,
      exemplar["solution"],
      exemplar["answer"]
    )

  def get_tabmwp_test_exemplars(self, exemplar):
    choices_str = "Please select from the following options: " + exemplar["choices"] \
                  if type(exemplar["choices"]) == str else ""
    return "Table: \n{}\nQuestion: {}\n{}".format(
      exemplar["table"],
      exemplar["question"],
      choices_str,
    )
    
    return string

  def get_aquarat_exemplars(self, exemplar):
    list_options = ast.literal_eval(exemplar["options"])
    options = "\n".join(list_options)
    return "Question: {}\nOptions:\n{}\nReasoning: {}. The correct option is {}.".format(
      exemplar["question"],
      options,
      exemplar["rationale"],
      exemplar["correct"]
    )

  def get_aquarat_test_exemplars(self, exemplar):
    list_options = ast.literal_eval(exemplar["options"])
    options = "\n".join(list_options)
    return "Question: {}\nOptions:\n{}\n".format(
      exemplar["question"],
      options,
    )
    
  def get_finqa_exemplars(self, exemplar):
    return "Read the following table, and then answer the question: \nTable:\n{}\nQuestion: {}\nEquation: {}\n. The answer is {}.".format(
      exemplar["table"],
      exemplar["question"],
      exemplar["program"],
      exemplar["answer"],
    )

  def get_finqa_test_exemplars(self, exemplar):
    return "Read the following table, and then answer the question: \nTable:\n{}\nQuestion: {}.".format(
      exemplar["table"],
      exemplar["question"],
    )

  def get_final_answer(self, text):
    pattern = r"Final Answer:\s*(.*)"
    match = re.search(pattern, text)
    if match:
      final_answer = match.group(1)
    else:
      raise Exception(f"No final answer for task: {self.task_idx}")

  def record_stats(self, output, prompt_len):
    def _update_stats(final_answer, real_answer):
      if final_answer is None:
        key = "FAILED"
      elif final_answer == real_answer:
        key = "CORRECT"
      else:
        key = "INCORRECT"
      self.stats[key] += 1

    final_answer = None
    output_lines = output[prompt_len:].splitlines()[::-1] # start backwards since final answer is more likely to be towards end
    if self.dataset == "strategyqa":
      for line in output_lines: 
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          if "yes" in lower_line:
            final_answer = "Yes"
          elif "no" in lower_line:
            final_answer = "No"
          if final_answer is not None:
            break
      real_answer = self.eval_df.iloc[self.task_idx]["answer"]
      # print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)
    elif self.dataset == "gsm8k":
      for line in output_lines:
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          final_answer = line.partition(":")[2].strip() # getting the half after the colon
        if final_answer is not None:
          break
      real_answer_raw = self.eval_df.iloc[self.task_idx]["answer"]
      matches = re.search(r"#+\s*(\d+)", real_answer_raw)
      real_answer = matches.group(1)
      # print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)
    elif self.dataset == "tabmwp":
      for line in output_lines:
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          final_answer = line.partition(":")[2].strip() # getting the half after the colon
        if final_answer is not None:
          break
      real_answer= self.eval_df.iloc[self.task_idx]["answer"]
      # print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)
    elif self.dataset == "aquarat":
      for line in output_lines:
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          final_answer = line.partition(":")[2].strip() # getting the half after the colon
        if final_answer is not None:
          break
      real_answer= self.eval_df.iloc[self.task_idx]["correct"]
      # print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)
    elif self.dataset == "finqa":
      for line in output_lines:
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          final_answer = line.partition(":")[2].strip() # getting the half after the colon
        if final_answer is not None:
          break
      real_answer= self.eval_df.iloc[self.task_idx]["answer"]
      # print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)