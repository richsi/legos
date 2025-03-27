import os
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
    exemplars: pd.DataFrame,
    num_reflections: int,
  ):
    # Default variables
    self.model = model
    self.benchmark = benchmark
    self.run_name = run_name
    self.exemplars = exemplars
    self.num_tasks = len(exemplars)
    self.num_reflections = num_reflections

    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.reflection_idx = 0           # Tracks current reflection index


  def run(self, reset: bool=True):
    """
    Starts new run
    """
    if reset:
      self.reset()

    while not self.done():
      self.step()

    utils.save_logs(self.benchmark, self.run_name, self.log_history)  # Once done with all tasks, save logs to txt file

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
        f"ID: {row['id']}\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}\n"
        f"Facts: {row['facts']}\n"
        f"Decomposition: {row['decomposition']}\n"
        f"Cluster: {row['cluster']}\n"
        f"Index: {row['index']}"
    )
    
    # Generate a simulated action (replace with your model's call as needed)
    action = f"Extracted insight from task {self.task_idx}"
    
    # Simulate an observation (this could be a result of processing or evaluation)
    observation = f"Processed task {self.task_idx} successfully."
    
    # Combine all elements into an experience log entry
    experience_log = (
        f"TASK {self.task_idx} Reflection {self.reflection_idx}\n"
        f"{experience_prompt}\n"
        f"Action: {action}\n"
        f"Observation: {observation}\n"
        "-------------------------------------"
    )
    
    # Save and print the experience log
    self.log_history.append(experience_log)
    print(experience_log)
    
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