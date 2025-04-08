import os
import re
import time
from typing import Callable
import pandas as pd
from src.agents.base import BaseAgent
import src.utils as utils

class TrainAgent(BaseAgent):
  def __init__(
    self,
    model: str,
    benchmark: str,
    run_name: str,
    **kwargs
  ):
    # Default variables
    self.model = model
    self.benchmark = benchmark
    self.run_name = run_name
    self.exemplars = kwargs["exemplars"]
    self.num_tasks = len(exemplars)

    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.stats = {"CORRECT": 0, "INCORRECT": 0}
    self.runtime = 0


  def run(self, reset: bool=True):
    """
    Starts new run
    """
    if reset:
      self.reset()

    start_time = time.time()

    while not self.done():
      # print(f"STARTING TASK {self.task_idx}\n")
      self.step()

    end_time = time.time()
    self.runtime = end_time - start_time

    utils.save_logs(
      self.benchmark, 
      self.run_name, 
      self.phase,
      self.log_history, 
      self.stats, 
      self.runtime
    )  # Once done with all tasks, save logs to txt file

  def step(self):
    """
    For the current task (CSV row), generate multiple reflections.
    Each reflection includes:
      - A header (e.g., "TASK X Reflection Y")
      - The staple prompt (which is part of the query to the LLM)
      - Task-specific details (from the CSV row)
      - A simulated chain-of-thought with interleaved thoughts and actions.
    """
    row = self.exemplars.iloc[self.task_idx]      # get current task
    # Build an experience prompt using the CSV fields
    experience_prompt = (
      # f"ID: {row['id']}\n"
      f"Facts: {row['facts']}\n"
      f"Question: {row['question']}\n"
      f"Answer: {row['answer']}\n"
      f"Decomposition: {row['decomposition']}\n"
      # f"Cluster: {row['cluster']}\n"
      # f"Index: {row['index']}"
    )

    # LLM api call to get model output
    llm_output = utils.query(self.model, self.benchmark, experience_prompt)
    result = self.compare_final_answer(llm_output) # returns CORRECT or INCORRECT

    self.stats[result] += 1 # increment final results
    
    # Combine all elements into an experience log entry
    experience_log = (
        f"TASK {self.task_idx}\n"
        f"Output: {llm_output}\n\n"
        f"Result: {result}\n"
        "-------------------------------------"
    )

    # print(experience_log)
    
    # Save and print the experience log
    self.log_history.append(experience_log)
    # Move to the next task
    self.task_idx += 1



  def done(self):
    """
    Checks if we are doing with the dataset
    """
    return self.task_idx >= self.num_tasks

  def reset(self):
    self.log_history = []
    self.task_idx = 0


  def compare_final_answer(self, task_text: str):
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