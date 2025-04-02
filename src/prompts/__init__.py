from . import strategyqa_prompts
from . import gsm8k_prompts

PROMPTS = dict(
  # StrategyQA
  StrategyQA_train=strategyqa_prompts.STRATEGYQA_PROMPT,
  StrategyQA_insight_extraction=strategyqa_prompts.STRATEGYQA_IE_PROMPT,
  StrategyQA_eval=strategyqa_prompts.STRATEGYQA_EVAL_PROMPT,

  # GSM8K
  GSM8K_insight_extraction=gsm8k_prompts.GSM8K_IE_PROMPT,
  GSM8K_eval=gsm8k_prompts.GSM8K_EVAL_PROMPT,
)