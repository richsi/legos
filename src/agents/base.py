from abc import ABC, abstractmethod

class BaseAgent(ABC):
  @abstractmethod
  def run(self):
    pass

  @abstractmethod
  def step(self):
    pass

  def done(self):
    return self.task_idx >= 20
  
  def reset(self):
    self.log_history = []
    self.task_idx = 0