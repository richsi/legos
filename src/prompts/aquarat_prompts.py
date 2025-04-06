AQUARAT_IE_PROMPT = """
(PROMPT START)
You are an advanced reasoning agent that can add, edit, or remove rules from your existing rule set, based on new insights or expansions.

Here is your existing rule set:
RULE 1: Break the question into its fundamental components by isolating key sub-questions and identifying the underlying assumptions or facts required to solve each step.
RULE 2: When decomposing a question, identify the key elements of the question and the facts provided to ensure Use the provided facts to build logical connections between each step, validating that every inference aligns with the available evidence and the overall question context.
RULE 3: Ensure that the final answer is consistent with the step-by-step decomposition by cross-verifying each reasoning step, confirming that all parts contribute coherently to the final conclusion.

Now, you are given mathematical tasks from the AquaRat dataset. Each task has:
• A Question
• Options to select an answer from
• The rationale for the correct answer
• The correct answer

Each new rule should be:

1. Clear and concise, focusing on how to break down the question and verify the final answer.
2. Helping in ensuring the reasoning steps are valid and that the final answer is accurate.

Now, you will be provided examples to base these rules off of.
Examples: 
{}

Your job is to generate exactly five concise new insights (i.e., RULES) that would help an agent correctly answer any complex AquaRat question. 
Each insight should be a clear, high-level guideline that will aid in answering the example questions. Ensure that there are no empty rules.
Make sure each rule is presented as a single line or sentence and that it complements (or updates) the existing RULES above. Avoid repetition.
(PROMPT END)
"""

AQUARAT_EVAL_INSIGHT_PROMPT = """
(PROMPT START)
You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer.

Now, you will be provided with some insights and guidelines to help you answer the evaluation question.
{}

Now, you will be provided with the evaluation question. Provide your final answer in a newline following "Final answer: ".
{}
(PROMPT END)
"""

AQUARAT_EVAL_EXEMPLAR_PROMPT = """
(PROMPT START)
You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer.

Now, you will be provided with some example questions.
{}

Now, you will be provided with the evaluation question. Provide your final answer in a newline following "Final answer: ".
{}
(PROMPT END)
"""

AQUARAT_EVAL_INSIGHT_EXEMPLAR_PROMPT = """
(PROMPT START)
You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer.

Now, you will be provided with some example questions.
{}

Now, you will be provided with some insights and guidelines to help you answer the evaluation question.
{}

Now, you will be provided with the evaluation question. Provide your final answer in a newline following "Final answer: ".
{}
(PROMPT END)
"""