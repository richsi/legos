GSM8K_IE_PROMPT = """
(TEMPLATE START)
You are an advanced reasoning agent that can add, edit, or remove rules from your existing rule set, based on new insights or expansions.

Here is your existing rule set:
RULE 1: Break the question into its fundamental components by isolating key sub-questions and identifying the underlying assumptions or facts required to solve each step.
RULE 2: When decomposing a question, identify the key elements of the question and the facts provided to ensure Use the provided facts to build logical connections between each step, validating that every inference aligns with the available evidence and the overall question context.
RULE 3: Ensure that the final answer is consistent with the step-by-step decomposition by cross-verifying each reasoning step, confirming that all parts contribute coherently to the final conclusion.

Now, you are given mathematical tasks from the GSM8K dataset. Each task has:
• A Question
• An Answer

Each new rule should be:

1. Clear and concise, focusing on how to break down the question and verify the final answer.
2. Helping in ensuring the reasoning steps are valid and that the final answer is accurate.

Now, you will be provided examples to base these rules off of.
Examples: 
{}

Your job is to generate exactly five concise new insights (i.e., RULES) that would help an agent correctly answer any complex GSM8K question. 
Each insight should be a clear, high-level guideline that will aid in answering the example questions. Ensure that there are no empty rules.
Make sure each rule is presented as a single line or sentence and that it complements (or updates) the existing RULES above. Avoid repetition.
(TEMPLATE END)
"""

GSM8K_EVAL_PROMPT = """
(TEMPLATE START)
Follow given example and solve the Test Question at the end in a similar manner by giving step by step reasoning followed by the Final Answer.
Below are some insights to guide you.
{}

Now, you will be given an example:
Question: Charles can earn $15 per hour when he housesits and $22 per hour when he walks a dog. If he housesits for 10 hours and walks 3 dogs, how many dollars will Charles earn?
Housesitting = 15 * 10 = 150
Dog walking = 22 * 3 = 66
Total earned is 150 + 66 = $216
Final Answer: 216

Following the example, generate step by step reasoning in 'Answer' and generate 'Final Answer' for the below question.

{}
(TEMPLATE END)
"""