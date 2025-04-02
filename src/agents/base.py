from abc import ABC, abstractmethod

class BaseAgent(ABC):
  @abstractmethod
  def run(self):
    pass

  @abstractmethod
  def step(self):
    pass

  @abstractmethod
  def done(self):
    pass
  
  def reset(self):
    self.log_history = []
    self.task_idx = 0