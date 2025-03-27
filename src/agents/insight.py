import pandas as pd
from src.agents.base import BaseAgent

class InsightAgent(BaseAgent):
  def __init__(
    self,
    model: str,
    benchmark: str,
    run_name: str,
    exemplars: pd.DataFrame
  ):
    self.model = model
    self.benchmark = benchmark
    self.run_name = run_name
    self.tasks_left = len(exemplars)


  def run(self):
    """
    Continues to step until tasks are done.
    """
    while not self.done():
      self.step()


  def step(self):
    """
    Extracts insight at each step
    """
    print(self.tasks_left)
    self.tasks_left -= 1
  

  def done(self):
    """
    Checks if we are doing with the dataset
    """
    return True if self.tasks_left == 0 else False