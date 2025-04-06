import os
import time
import pandas as pd
import re
from src.agents.base import BaseAgent
import src.utils as utils
import ast
from src.models import QUERY

class InsightAgent(BaseAgent):
  def __init__(self, **kwargs):
    # Default variables
    self.model = kwargs["model"]
    self.phase = kwargs["phase"]
    self.dataset = kwargs["dataset"]
    self.run_name = kwargs["run_name"]
    self.exemplars = pd.read_csv(os.path.join(os.getenv(kwargs["dataset"].upper()), kwargs["train"])) # pd.DataFrame type

    self.num_tasks = len(kwargs["train"])
    self.log_history = []          
    self.task_idx = 0                 # Tracks current task index 
    self.runtime = 0
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
      self.dataset, 
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

    exemplar_getter = {
      "strategyqa": self.get_strategyqa_exemplars,
      "gsm8k": self.get_gsm8k_exemplars,
      "tabmwp": self.get_tabmwp_exemplars,
    }.get(self.dataset)

    if exemplar_getter is None:
      raise ValueError(f"Unsupported dataset: {self.dataset}")

    all_exemplars = [exemplar_getter(row) for _, row in self.exemplars.iterrows()]
    exemplars = "\n\n".join(all_exemplars)

    # LLM api call to get model output
    kwargs = dict(exemplars=exemplars)
    formatted_prompt = utils.format_prompt(self.phase, self.dataset, **kwargs) # formatting the prompt
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

  def done(self):
    """
    Checks if we are doing with the dataset
    """
    return self.task_idx >= self.num_tasks

  def reset(self):
    self.log_history = []
    self.task_idx = 0