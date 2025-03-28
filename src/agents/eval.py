import time
import pandas as pd
from src.agents.base import BaseAgent
import src.utils as utils
import ast

class EvalAgent(BaseAgent):
  def __init__(
    self,
    model: str,
    phase: str,
    benchmark: str,
    run_name: str,
    exemplars: pd.DataFrame,
  ):
    # Default variables
    self.model = model
    self.phase = phase
    self.benchmark = benchmark
    self.run_name = run_name
    self.exemplars = exemplars
    self.num_tasks = len(exemplars)

    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.runtime = 0
    self.gpu_usage = []
    self.stats = None


  def run(self, reset: bool=True):
    """
    Starts new run
    """
    if reset:
      self.reset()

    start_time = time.time()

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
      - The staple task description prompt
      - Exemplar details (from the CSV)
      - Generated insights from the exemplars
    """

    all_test_exemplars = []
    for idx, row in self.exemplars.iterrows(): # ensure the exemplars are from the eval dataset
      all_test_exemplars.append(self.get_prompt_test(row))
    prompt = "\n---\n".join(all_test_exemplars)

    # LLM api call to get model output
    llm_output = utils.query(self.model, self.phase, self.benchmark, prompt)

    # Combine all elements into an experience log entry
    experience_log = (
        f"Extracted Insights\n"
        f"Output: {llm_output}\n\n"
        "-------------------------------------"
    )

    print(experience_log)
    
    # Save and print the experience log
    self.log_history.append(experience_log)

  
  def get_prompt_test(self, exemplar):
    string = "Facts: "
    for fact in exemplar["facts"]:
      string += fact
    string += "\nQuestion: " + exemplar["question"]
    return string


  def done(self):
    """
    Checks if we are doing with the dataset
    """
    return self.task_idx >= self.num_tasks

  def reset(self):
    self.log_history = []
    self.task_idx = 0