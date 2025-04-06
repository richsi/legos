from . import strategyqa_prompts, gsm8k_prompts, tabmwp_prompts, aquarat_prompts, finqa_prompts

PROMPTS = dict(
  # StrategyQA
  StrategyQA_insight_extraction=    strategyqa_prompts.STRATEGYQA_IE_PROMPT,
  StrategyQA_eval_insight=          strategyqa_prompts.STRATEGYQA_EVAL_INSIGHT_PROMPT,
  StrategyQA_eval_exemplar=         strategyqa_prompts.STRATEGYQA_EVAL_EXEMPLAR_PROMPT,
  StrategyQA_eval_insight_exemplar= strategyqa_prompts.STRATEGYQA_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # GSM8K
  GSM8K_insight_extraction=    gsm8k_prompts.GSM8K_IE_PROMPT,
  GSM8K_eval_insight=          gsm8k_prompts.GSM8K_EVAL_INSIGHT_PROMPT,
  GSM8K_eval_exemplar=         gsm8k_prompts.GSM8K_EVAL_EXEMPLAR_PROMPT,
  GSM8K_eval_insight_exemplar= gsm8k_prompts.GSM8K_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # TabMWP
  TabMWP_insight_extraction=    tabmwp_prompts.GSM8K_IE_PROMPT,
  TabMWP_eval_insight=          tabmwp_prompts.GSM8K_EVAL_INSIGHT_PROMPT,
  TabMWP_eval_exemplar=         tabmwp_prompts.GSM8K_EVAL_EXEMPLAR_PROMPT,
  TabMWP_eval_insight_exemplar= tabmwp_prompts.GSM8K_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # AquaRat
  AquaRat_insight_extraction=    aquarat_prompts.AQUARAT_IE_PROMPT,
  AquaRat_eval_insight=          aquarat_prompts.AQUARAT_EVAL_INSIGHT_PROMPT,
  AquaRat_eval_exemplar=         aquarat_prompts.AQUARAT_EVAL_EXEMPLAR_PROMPT,
  AquaRat_eval_insight_exemplar= aquarat_prompts.AQUARAT_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # FinQA
  FinQA_insight_extraction=    finqa_prompts.FINQA_IE_PROMPT,
  FinQA_eval_insight=          finqa_prompts.FINQA_EVAL_INSIGHT_PROMPT,
  FinQA_eval_exemplar=         finqa_prompts.FINQA_EVAL_EXEMPLAR_PROMPT,
  FinQA_eval_insight_exemplar= finqa_prompts.FINQA_EVAL_INSIGHT_EXEMPLAR_PROMPT,
)