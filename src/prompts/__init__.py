from . import strategyqa_prompts, gsm8k_prompts, tabmwp_prompts, aquarat_prompts, finqa_prompts

PROMPTS = dict(
  # StrategyQA
  strategyqa_insight_extraction=    strategyqa_prompts.STRATEGYQA_IE_PROMPT,
  strategyqa_eval_insight=          strategyqa_prompts.STRATEGYQA_EVAL_INSIGHT_PROMPT,
  strategyqa_eval_exemplar=         strategyqa_prompts.STRATEGYQA_EVAL_EXEMPLAR_PROMPT,
  strategyqa_eval_insight_exemplar= strategyqa_prompts.STRATEGYQA_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # GSM8K
  gsm8k_insight_extraction=    gsm8k_prompts.GSM8K_IE_PROMPT,
  gsm8k_eval_insight=          gsm8k_prompts.GSM8K_EVAL_INSIGHT_PROMPT,
  gsm8k_eval_exemplar=         gsm8k_prompts.GSM8K_EVAL_EXEMPLAR_PROMPT,
  gsm8k_eval_insight_exemplar= gsm8k_prompts.GSM8K_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # TabMWP
  tabmwp_insight_extraction=    tabmwp_prompts.TABMWP_IE_PROMPT,
  tabmwp_eval_insight=          tabmwp_prompts.TABMWP_EVAL_INSIGHT_PROMPT,
  tabmwp_eval_exemplar=         tabmwp_prompts.TABMWP_EVAL_EXEMPLAR_PROMPT,
  tabmwp_eval_insight_exemplar= tabmwp_prompts.TABMWP_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # AquaRat
  aquarat_insight_extraction=    aquarat_prompts.AQUARAT_IE_PROMPT,
  aquarat_eval_insight=          aquarat_prompts.AQUARAT_EVAL_INSIGHT_PROMPT,
  aquarat_eval_exemplar=         aquarat_prompts.AQUARAT_EVAL_EXEMPLAR_PROMPT,
  aquarat_eval_insight_exemplar= aquarat_prompts.AQUARAT_EVAL_INSIGHT_EXEMPLAR_PROMPT,

  # FinQA
  finqa_insight_extraction=    finqa_prompts.FINQA_IE_PROMPT,
  finqa_eval_insight=          finqa_prompts.FINQA_EVAL_INSIGHT_PROMPT,
  finqa_eval_exemplar=         finqa_prompts.FINQA_EVAL_EXEMPLAR_PROMPT,
  finqa_eval_insight_exemplar= finqa_prompts.FINQA_EVAL_INSIGHT_EXEMPLAR_PROMPT,
)