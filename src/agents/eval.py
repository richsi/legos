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
    model: str,
    phase: str,
    benchmark: str,
    run_name: str,
    **kwargs
  ):
    # Default variables
    self.model = model
    self.phase = phase
    self.benchmark = benchmark
    self.run_name = run_name
    self.train_exemplars = kwargs["exemplars"]
    self.num_tasks = 0

    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.runtime = 0
    self.gpu_usage = []
    self.stats = {"CORRECT": 0, "INCORRECT": 0}

    self.eval_file = kwargs["eval"]
    self.eval_df = None
    self.all_test_exemplars = []


  def run(self, reset: bool=True):
    """
    Starts new run
    """
    if reset:
      self.reset()

    all_train_exemplars = []

    for idx, row in self.train_exemplars.iterrows(): # ensure the exemplars are from the train dataset
      all_train_exemplars.append(self.get_prompt(row))

    eval_path = os.path.join(os.getenv("DATA"), self.benchmark.lower(), self.eval_file)
    self.eval_df = pd.read_csv(eval_path)
    for idx, row in self.eval_df.iterrows(): # ensure the exemplars are from the eval dataset
      self.all_test_exemplars.append(self.get_prompt_test(row))
    self.num_tasks = len(self.all_test_exemplars) # setting num_tasks for task_idx

    train_data = "\n---\n".join(all_train_exemplars)
    # LLM api call to get model output
    insights = utils.get_insights(self.model, self.benchmark, self.run_name)

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

    EM = self.stats["CORRECT"] / (self.stats["CORRECT"] + self.stats["INCORRECT"]) * 100 

    results_dict = {
      "matches": [self.stats["CORRECT"]],
      "mismatches": [self.stats["INCORRECT"]],
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
    )  # Once done with all tasks, save logs to txt file


  def step(self, **kwargs):
    """
    Log:
      - The static task description prompt
      - Exemplar details (from the CSV)
      - Generated insights from the exemplars
      - Final evaluation answers
    """

    batch_prompts = []
    batch_real_answers = []
    batch_indices = []

    # batching the data
    for i in range(kwargs["batch_size"]):
      if self.done():
        break
      batch_prompts.append(self.all_test_exemplars[self.task_idx])
      batch_real_answers.append(self.eval_df.iloc[self.task_idx]["answer"])
      batch_indices.append(self.task_idx)
      self.task_idx += 1

    kwargs["test_data"] = "\n---\n".join(batch_prompts)
    # print(kwargs["test_data"])

    # kwargs["test_data"] = self.all_test_exemplars[self.task_idx]
    prompt = utils.format_prompt(self.phase, self.benchmark, **kwargs)

    llm_output = utils.query(self.model, prompt)

    print("SPLICED:\n",llm_output[len(prompt):])

    batch_final_answers = re.findall(r"^Final Answer:\s(.*?)(?:\.)?$", llm_output[len(prompt):], re.MULTILINE)
    print("\nFinal answers: ", batch_final_answers)

    # recording stats
    print("Real answers: ", batch_real_answers)
    for final_answer, real_answer in zip(batch_final_answers, batch_real_answers):
      key = "CORRECT" if final_answer.strip() == real_answer.strip() else "INCORRECT"
      self.stats[key] += 1

    # Combine all elements into an experience log entry
    experience_log = (
        f"{self.model} Tasks {batch_indices}:\n{llm_output}\n\n"
        "-------------------------------------"
    )

    # Save and print the experience log
    self.log_history.append(experience_log)

  def get_prompt(self, exemplar):
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
  
  def get_prompt_test(self, exemplar):
    string = "Facts: "
    for fact in exemplar["facts"]:
      string += fact
    string += "\nQuestion: " + exemplar["question"]
    return string

  def get_final_answer(self, text):
    pattern = r"Final Answer:\s*(.*)"
    match = re.search(pattern, text)
    if match:
      final_answer = match.group(1)
    else:
      raise Exception(f"No final answer for task: {self.task_idx}")