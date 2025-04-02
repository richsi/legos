import time
import pandas as pd
from src.agents.base import BaseAgent
import src.utils as utils
import ast
from src.models import QUERY

class InsightAgent(BaseAgent):
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
    self.exemplars = kwargs["exemplars"]
    self.num_tasks = len(kwargs["exemplars"])

    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.reflection_idx = 0           # Tracks current reflection index
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
      self.model,
      self.benchmark, 
      self.run_name, 
      self.phase,
      self.log_history, 
      self.stats, 
      self.runtime
    )  # Once done with all tasks, save logs to txt file

  def step(self):
    """
    Log:
      - The static task description prompt
      - Exemplar details (from the CSV)
      - Generated insights from the exemplars
    """

    all_exemplars = []
    for idx, row in self.exemplars.iterrows():
      all_exemplars.append(self.get_prompt(row))
    exemplars = "\n---\n".join(all_exemplars)

    # LLM api call to get model output
    kwargs = dict(exemplars=exemplars)
    formatted_prompt = utils.format_prompt(self.phase, self.benchmark, **kwargs) # formatting the prompt
    llm_output = QUERY[self.model](formatted_prompt) # querying the LLM model
    print(llm_output)

    # Combine all elements into an experience log entry
    experience_log = (
        f"Extracted Insights\n"
        f"Output: {llm_output}\n\n"
        "-------------------------------------"
    )

    print(experience_log)
    
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


  def done(self):
    """
    Checks if we are doing with the dataset
    """
    return self.task_idx >= self.num_tasks

  def reset(self):
    self.log_history = []
    self.task_idx = 0