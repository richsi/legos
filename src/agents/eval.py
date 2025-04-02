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
    self.benchmark = kwargs["benchmark"]
    self.run_name = kwargs["run_name"]
    self.train_exemplars = pd.read_csv(os.path.join(os.getenv(kwargs["benchmark"].upper()), kwargs["train"])) # pd.DataFrame type

    self.num_tasks = 0
    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.runtime = 0
    self.stats = {"CORRECT": 0, "INCORRECT": 0, "FAILED":0}

    self.eval_file = kwargs["eval"]
    self.eval_df = None
    self.all_test_exemplars = []


  def run(self, reset: bool=True):
    """
    Starts new run
    """
    if reset:
      self.reset()

    # Dict mapping to correct exemplar getter function
    exemplar_getter = {
      "StrategyQA": self.get_strategyqa_exemplars,
      "GSM8K": self.get_gsm8k_exemplars,
      "TabMWP": self.get_tabmwp_exemplars,
    }.get(self.benchmark)

    # Check that benchmark is valid and function exists
    if exemplar_getter is None:
      raise ValueError(f"Unsupported benchmark: {self.benchmark}")

    # Formatting the exemplar to pass into prompt
    all_train_exemplars = [exemplar_getter(row) for _, row in self.train_exemplars.iterrows()]
    train_data = "\n---\n".join(all_train_exemplars)

    # Getting the eval file path
    eval_path = os.path.join(os.getenv(self.benchmark.upper()), self.eval_file)
    # Getting eval file contents
    self.eval_df = pd.read_csv(eval_path)

    # Dict mapping to correct test exemplar getter function
    test_exemplar_getter = {
      "StrategyQA": self.get_strategyqa_test_exemplars,
      "GSM8K": self.get_gsm8k_exemplars,
      "TabMWP": self.get_tabmwp_exemplars,
    }.get(self.benchmark)

    if test_exemplar_getter is None:
      raise ValueError(f"Unsupported benchmark: {self.benchmark}")

    # Formatting the test exemplars to pass into prompt
    self.all_test_exemplars = [test_exemplar_getter(row) for _, row in self.eval_df.iterrows()]
    # Assigning num tasks
    self.num_tasks = len(self.all_test_exemplars) 

    # Getting insights from same run_name from logs/insight_extraction
    insights = utils.get_insights(self.model, self.benchmark, self.run_name)
    print(f"INSIGHTS:\n{insights}")

    kwargs = dict(
      exemplars=train_data,
      insights=insights,
      batch_size=5
    )

    start_time = time.time()
    while not self.done():
      print(f"STARTING TASK {self.task_idx}\n")
      self.step(**kwargs)
    end_time = time.time()
    self.runtime = end_time - start_time

    EM = self.stats["CORRECT"] / (self.stats["CORRECT"] + self.stats["INCORRECT"] + self.stats["FAILED"]) * 100 

    results_dict = {
      "matches": [self.stats["CORRECT"]],
      "mismatches": [self.stats["INCORRECT"] + self.stats["FAILED"]],
      "EM": [EM],
      "train_data_len": [len(all_train_exemplars)],
      "test_data_len": [len(self.all_test_exemplars)],
      "model": [self.model],
      "dataset": [self.benchmark],
      "run_name": [self.run_name]
    }

    utils.save_logs(
      self.model,
      self.benchmark, 
      self.run_name, 
      self.phase,
      self.log_history, 
      self.stats, 
      self.runtime,
      results_dict
    )  


  def step(self, **kwargs):
    """
    Log:
      - The static task description prompt
      - Exemplar details (from the CSV)
      - Generated insights from the exemplars
      - Final evaluation answers
    """
    # batch_prompts = []
    # batch_real_answers = []
    # batch_indices = []
    # batching the data
    # for i in range(kwargs["batch_size"]):
    #   if self.done():
    #     break
    #   kwargs["test_data"] = self.all_test_exemplars[self.task_idx]
    #   subprompt = utils.format_prompt(self.phase, self.benchmark, **kwargs)
    #   batch_prompts.append(subprompt)
    #   batch_real_answers.append(self.eval_df.iloc[self.task_idx]["answer"])
    #   batch_indices.append(self.task_idx)
    #   self.task_idx += 1

    kwargs["test_data"] = self.all_test_exemplars[self.task_idx]
    # Formatting prompt for LLM
    prompt = utils.format_prompt(self.phase, self.benchmark, **kwargs) 
    # Querying the LLM
    llm_output = utils.query(self.model, prompt)
    # print("SPLICED:\n",llm_output[len(prompt):])
    # Recording the stats
    self.record_stats(llm_output, len(prompt))
    # Combine all elements into an experience log entry
    experience_log = (
        f"{self.model} Task {self.task_idx}:\n{llm_output}\n\n"
        "-------------------------------------"
    )
    # Save and print the experience log
    self.log_history.append(experience_log)
    # increment task index
    self.task_idx += 1

  def done(self):
    # return self.task_idx >= self.num_tasks
    return self.task_idx >= 5

  def get_strategyqa_exemplars(self, exemplar):
    string = "Facts: "
    for fact in exemplar["facts"]:
      string += fact
    string += "\nQuestion: " + exemplar["question"]
    string += "\nAnswer:\n"
    decomp_list = ast.literal_eval(exemplar["decomposition"])
    for idx, question in enumerate(decomp_list):
      string += "Sub-question {}: {}\n".format(idx+1, question)
    string += "The answer is: " + exemplar["answer"]
    return string
  
  def get_strategyqa_test_exemplars(self, exemplar):
    string = "Facts: "
    for fact in exemplar["facts"]:
      string += fact
    string += "\nQuestion: " + exemplar["question"]
    return string

  def get_gsm8k_exemplars(self, exemplar):
    answer = re.sub("<<.*?>>", "", exemplar["answer"])
    answer = answer.replace("####", "Final Answer:")
    string = "Question: " + exemplar["question"] + "\n" + answer + "\n\n"
    return string

  def get_tabmwp_exemplars(self, exemplar):
    choices_str = "Please select from the following options: " + exemplar["choices"] \
                  if type(exemplar["choices"]) == str else ""
    string = "\nTable:\n" + exemplar["table"] + "\nQuestion:" + exemplar["question"] \
              + choices_str + "\nAnswer:" + exemplar["solution"] + "\nThe answer is:" \
              + exemplar["answer"]
    return string

  def get_final_answer(self, text):
    pattern = r"Final Answer:\s*(.*)"
    match = re.search(pattern, text)
    if match:
      final_answer = match.group(1)
    else:
      raise Exception(f"No final answer for task: {self.task_idx}")

  def record_stats(self, output, prompt_len):
    final_answer = None
    output_lines = output[prompt_len:].splitlines()[::-1] # start backwards since final answer is more likely to be towards end

    print("OUTPUT_LINES: ", output_lines)

    def _update_stats(final_answer, real_answer):
      if final_answer is None:
        key = "FAILED"
      elif final_answer == real_answer:
        key = "CORRECT"
      else:
        key = "INCORRECT"
      self.stats[key] += 1

    if self.benchmark == "StrategyQA":
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
      print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)
    elif self.benchmark == "GSM8K":
      for line in output_lines:
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          final_answer = line.partition(":")[2].strip() # getting the half after the colon
        if final_answer is not None:
          break
      real_answer_raw = self.eval_df.iloc[self.task_idx]["answer"]
      matches = re.search(r"####\s*(\d+)", real_answer_raw)
      real_answer = matches.group(1)
      print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)
    elif self.benchmark == "TabMWP":
      for line in output_lines:
        lower_line = line.lower()
        if "final" in lower_line and "answer" in lower_line:
          final_answer = line.partition(":")[2].strip() # getting the half after the colon
        if final_answer is not None:
          break
      real_answer= self.eval_df.iloc[self.task_idx]["answer"]
      print(f"\nFinal Answer: {final_answer}\nReal Answer: {real_answer}")
      _update_stats(final_answer, real_answer)