from .base import BaseAgent
from .train import TrainAgent
from .insight import InsightAgent
from .eval import EvalAgent

AGENT = dict(
  train=TrainAgent,
  insight_extraction=InsightAgent, 
  eval=EvalAgent
)