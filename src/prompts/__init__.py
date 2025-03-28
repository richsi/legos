from . import strategyqa_prompts

PROMPTS = dict(
  # StrategyQA
  StrategyQA_train=strategyqa_prompts.STRATEGYQA_PROMPT,
  StrategyQA_insight_extraction=strategyqa_prompts.STRATEGYQA_IE_PROMPT,
  StrategyQA_eval=strategyqa_prompts.STRATEGYQA_EVAL_PROMPT,

  # GSM8K
)