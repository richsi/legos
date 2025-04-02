STRATEGYQA_PROMPT = """(TEMPLATE START)
You are a concise, multi-step question answering system. For each task, you are provided with three fields: a Question, a list of Facts, and a Decomposition that breaks the Question into multiple reasoning steps. Your job is to use the Decomposition to structure your reasoning and to use the Facts to support your answer.

For example:
Task:
Question: Has Drew Carey outshined Doug Davidson's tenure?
Facts: ["Drew Carey has been the host of the Price is Right for over 13 years.", "Doug Davidson hosted the Price is Right from 1994-1995."]
Decomposition: ["How long has Drew Carey hosted the Price is Right?", "How long did Doug Davidson host the Price is Right?", "Is #1 longer than #2?"]

Expected Output:
Thought: The decomposition shows that Drew Carey’s tenure is significantly longer than Doug Davidson's. The Facts confirm Drew Carey hosted for over 13 years while Doug Davidson hosted for only about one year.
Final Answer: Yes

Now, using the provided Question, Facts, and Decomposition fields, answer the question following the above format.

When responding, your output must contain only two lines exactly in the format below:

Thought: <your concise reasoning here, incorporating the decomposition steps and the facts>
Final Answer: <your concise final answer here>
(TEMPLATE END)

"""

STRATEGYQA_IE_PROMPT = """
(TEMPLATE START)
You are an advanced reasoning agent that can add, edit, or remove rules from your existing rule set, based on new insights or expansions.

Here is your existing rule set:
RULE 1: Break the question into its fundamental components by isolating key sub-questions and identifying the underlying assumptions or facts required to solve each step.
RULE 2: When decomposing a question, identify the key elements of the question and the facts provided to ensure Use the provided facts to build logical connections between each step, validating that every inference aligns with the available evidence and the overall question context.
RULE 3: Ensure that the final answer is consistent with the step-by-step decomposition by cross-verifying each reasoning step, confirming that all parts contribute coherently to the final conclusion.
...
-----
Now, you are given a StrategyQA task. Each task has:
• A Question
• A set of Facts
• An Answer
• A Decomposition (a step-by-step outline of how the question might be solved)

Your job is to generate between 5 and 15 concise new insights (i.e., rules or directions) that would help an agent correctly answer any complex StrategyQA question. Each new rule should be:

1. Generally applicable to tasks involving a Question, Facts, Answer, and Decomposition.
2. Clear and concise, focusing on how to break down the question, leverage the facts, and verify the final answer.
3. Helpful in ensuring the reasoning steps are valid and that the final answer is accurate.

Examples:
(START OF EXAMPLES)
{}
(END OF EXAMPLES)

Based on the provided task description and examples, generate between 5 and 15 new or updated rules.
Each insight should be a clear, high-level guideline that strengthens the reasoning process.
Ensure that there are no empty rules.
Make sure each rule is presented as a single line or sentence and that it complements (or updates) the existing rules above. Avoid repetition.
Please provide these new or revised rules below in the format: RULE <number>: <rule>, starting with number 1.
(TEMPLATE END)
"""


# Note batch size
STRATEGYQA_EVAL_PROMPT = """
(TEMPLATE START)
You are a reasoning and insight generation assistant for multi-step question answering tasks. Your primary role is to systematically guide the reasoning process and arrive at the correct answer. You will be provided with an evaluation question. The question comes with its set of supporting facts.
Here are extracted rules and guidelines from the provided examples to ensure proper reasoning:
{}

Your response for the question in the batch must be formatted exactly as follows:
Question: <The original test question>
Thoughts: <Your concise, structured reasoning here>
Final Answer: <Yes or No>

Now, you will be provided with the evaluation question and fact(s) below.
{}
(TEMPLATE END)
"""

# STRATEGYQA_EVAL_PROMPT = """
# (TEMPLATE START)

# You are a reasoning and insight generation assistant for multi-step question answering tasks. Your primary role is to systematically guide the reasoning process and arrive at the correct answer. You will be provided with a batch of evaluation questions. Each question in the batch comes with its set of supporting facts (and may include an answer field in guided examples) but does NOT include sub-questions. For each question, your task is to:

# 1. Review the provided facts carefully.
# 2. Analyze the test question using these facts.
# 3. Clearly outline your reasoning strategy in a section labeled "Thoughts:".
# 4. Conclude with a final answer in a section labeled "Final Answer:" that is either "Yes" or "No".

# Your response for each question in the batch must be formatted exactly as follows, in the same order as the input:

# Question: [The original test question]
# Thoughts: [Your concise, structured reasoning here]
# Final Answer: Yes/No

# Guided example with reasoning strategy and output format:

# Question: Is latitude required to determine the coordinates of an area?
# Facts: ["Longitude is required for determining coordinates.", "Latitude is also required to determine coordinates of an area."]
# Reasoning Strategy:
# - Identify the key data points provided by the facts.
# - Confirm that both longitude and latitude are mentioned as necessary for determining coordinates.
# - Conclude that the presence of latitude is required.
# Final Answer Validation:
# - Ensure that the reasoning directly addresses the test question and aligns with the facts provided.
# Output for this question:
# Question: Is latitude required to determine the coordinates of an area?
# Thoughts: [Your reasoning based on the provided facts]
# Final Answer: Yes/No

# (END OF GUIDED EXAMPLES)

# Unguided examples (these examples include an 'Answer' field, but you must ignore it and produce your own output in the required format):
# {}

# (END OF UNGUIDED EXAMPLES)

# Here are extracted insights and guidelines from the provided examples to ensure proper reasoning:
# {}

# Now, you will be provided with a batch of evaluation questions below. For each question in the batch, produce your answer in the following format:

# Question: [the original test question]
# Thoughts: [whatever thoughts or reasoning steps are needed to arrive at your conclusion]
# Final Answer: Yes/No

# Ensure that your answers are provided in the same order as the questions appear in the input. 
# Ensure that each of the questions below gets answered.
# {}
# ---
# (TEMPLATE END)
# """
