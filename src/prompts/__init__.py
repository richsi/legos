from . import strategyqa_prompts, gsm8k_prompts, tabmwp_prompts

PROMPTS = dict(
  # StrategyQA
  StrategyQA_train=strategyqa_prompts.STRATEGYQA_PROMPT,
  StrategyQA_insight_extraction=strategyqa_prompts.STRATEGYQA_IE_PROMPT,
  StrategyQA_eval=strategyqa_prompts.STRATEGYQA_EVAL_PROMPT,

  # GSM8K
  GSM8K_insight_extraction=gsm8k_prompts.GSM8K_IE_PROMPT,
  GSM8K_eval=gsm8k_prompts.GSM8K_EVAL_PROMPT,

  # TabMWP
  TabMWP_insight_extraction=tabmwp_prompts.TABMWP_IE_PROMPT,
  TabMWP_eval=tabmwp_prompts.TABMWP_EVAL_PROMPT,
)