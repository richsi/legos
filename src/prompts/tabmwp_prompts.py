TABMWP_IE_PROMPT = """
(TEMPLATE START)
You are an advanced reasoning agent that can add, edit, or remove rules from your existing rule set, based on new insights or expansions.

Here is your existing rule set:
RULE 1: Break the question into its fundamental components by isolating key sub-questions and identifying the underlying assumptions or facts required to solve each step.
RULE 2: When decomposing a question, identify the key elements of the question and the facts provided to ensure Use the provided facts to build logical connections between each step, validating that every inference aligns with the available evidence and the overall question context.
RULE 3: Ensure that the final answer is consistent with the step-by-step decomposition by cross-verifying each reasoning step, confirming that all parts contribute coherently to the final conclusion.

Now, you are given tasks from the TabMWP dataset. Each task has:
• A table for context
• The Question
• Multiple choices to select an answer from
• An answer containing the reasoning that leads to the correct answer
• A concise final answer

Your job is to generate between 5 and 15 concise new insights (i.e., rules or directions) that would help an agent correctly answer any complex TabMWP question. Each new rule should be:
Examples: 
{}


Based on the provided task description and examples, generate exactly 10 new or updated rules.
Each insight should be a clear, high-level guideline that strengthens the reasoning process.
Ensure that there are no empty rules.
Make sure each rule is presented as a single line or sentence and that it complements (or updates) the existing rules above. Avoid repetition.
(TEMPLATE END)
"""

TABMWP_EVAL_PROMPT = """
(TEMPLATE START)
Follow each of the given examples by using its Table to find the answer for its Question with the reasoning after Answer: and final answer after The answer is:.

Examples:
{}


Following the given examples, generate step by step reasoning in 'Answer' and generate 'Final Answer' for the below question.

{}
(TEMPLATE END)
"""