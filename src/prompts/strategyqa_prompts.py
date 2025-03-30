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
You are an advanced reasoning agent that can add, edit, or remove rules from your existing rule set, based on new insights or expansions.

Here is your existing rule set:
RULE 1: <EXISTING RULE TEXT>
RULE 2: <EXISTING RULE TEXT>
RULE 3: <EXISTING RULE TEXT>
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

Please provide these new or revised rules below. Make sure each rule is presented as a single line or sentence and that it complements (or updates) the existing rules above.

Examples:
{}

(END OF EXAMPLES)

Based on the provided task description and examples, generate between 5 and 15 new or updated rules.
By examining the list of existing rules, you can perform the following operations: add, edit, downvote, or upvote so that the new rules are GENERAL and HIGH LEVEL insights of the trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future.
Have an emphasis on tips that help the agent by identifying key elements from the Question, leveraging the Facts, examining the Decomposition, and ensuring the Answer is correct. Each insight should be a clear, high-level guideline that strengthens the reasoning process.
At the end, ensure that there are no empty rules.
"""

# STRATEGYQA_IE_PROMPT = """
# (TEMPLATE START)
# You are an insight generation assistant designed to provide structured guidance on solving multi-step, complex reasoning tasks from the StrategyQA dataset. Your primary goal is to generate generalizable insights and reasoning strategies that can be broadly applied to accurately solve similar reasoning problems. 

# Given a set of examples, each containing:
# - A Question
# - Relevant Facts
# - A Decomposition into sub-questions
# - The correct Answer

# your task is to:

# 1. Identify general reasoning strategies from how the provided facts support the correct answer.
# 2. Outline best practices for decomposing complex questions effectively.
# 3. Highlight methods to systematically verify the accuracy of intermediate reasoning steps.
# 4. Provide guidelines for effectively utilizing provided facts to construct logical pathways to the correct answer.

# You should NOT provide example-specific conclusions or directly state the answer. Instead, your insights should help establish clear rules and frameworks that guide effective reasoning.

# Examples:

# Question: Is latitude required to determine the coordinates of an area?
# Facts: ["Longitude is required for determining coordinates.", "Latitude is also required to determine coordinates of an area."]
# Decomposition: ["What are the two data points needed to determine coordinates?", "Is latitude one of these data points?"]
# Answer: Yes

# Insight Generation:
# - Always clearly identify all required data points when asked about coordinate determination.
# - Verify explicitly if each potential data point is listed in the provided facts.
# - Ensure each sub-question directly references or is addressed by the given facts.

# Question: Has Drew Carey outshined Doug Davidson's tenure?
# Facts: ["Drew Carey hosted The Price is Right for over 13 years.", "Doug Davidson hosted The Price is Right from 1994-1995."]
# Decomposition: ["How long has Drew Carey hosted The Price is Right?", "How long did Doug Davidson host The Price is Right?", "Is Drew Carey’s tenure longer?"]
# Answer: Yes

# Insight Generation:
# - When comparing durations, explicitly extract and quantify the time spans from the facts.
# - Clearly outline the comparison process, demonstrating a logical step-by-step evaluation.
# - Reinforce the importance of explicitly answering each sub-question before reaching a conclusion.

# (END OF EXAMPLES)

# Following this demonstrated approach, generate between 5 to 15 broadly applicable insights, outlining clear reasoning strategies, rules for decomposing complex questions, and guidelines for verifying accuracy and aligning reasoning steps with provided facts for the examples below.

# {}
# (TEMPLATE END)
# """

STRATEGYQA_EVAL_PROMPT = """
(TEMPLATE START)

You are a reasoning and insight generation assistant for multi-step question answering tasks. Your primary role is to systematically guide the reasoning process and arrive at the correct answer. You will be provided with a batch of evaluation questions. Each question in the batch comes with its set of supporting facts (and may include an answer field in guided examples) but does NOT include sub-questions. For each question, your task is to:

1. Review the provided facts carefully.
2. Analyze the test question using these facts.
3. Clearly outline your reasoning strategy in a section labeled "Thoughts:".
4. Conclude with a final answer in a section labeled "Final Answer:" that is either "Yes" or "No".

Your response for each question in the batch must be formatted exactly as follows, in the same order as the input:

Question: [The original test question]
Thoughts: [Your concise, structured reasoning here]
Final Answer: Yes/No

Guided example with reasoning strategy and output format:

Question: Is latitude required to determine the coordinates of an area?
Facts: ["Longitude is required for determining coordinates.", "Latitude is also required to determine coordinates of an area."]
Reasoning Strategy:
- Identify the key data points provided by the facts.
- Confirm that both longitude and latitude are mentioned as necessary for determining coordinates.
- Conclude that the presence of latitude is required.
Final Answer Validation:
- Ensure that the reasoning directly addresses the test question and aligns with the facts provided.
Output for this question:
Question: Is latitude required to determine the coordinates of an area?
Thoughts: [Your reasoning based on the provided facts]
Final Answer: Yes/No

(END OF GUIDED EXAMPLES)

Unguided examples (these examples include an 'Answer' field, but you must ignore it and produce your own output in the required format):
{}

(END OF UNGUIDED EXAMPLES)

Here are extracted insights and guidelines from the provided examples to ensure proper reasoning:
{}

Now, you will be provided with a batch of evaluation questions below. For each question in the batch, produce your answer in the following format:

Question: [the original test question]
Thoughts: [whatever thoughts or reasoning steps are needed to arrive at your conclusion]
Final Answer: Yes/No

Ensure that your answers are provided in the same order as the questions appear in the input.
{}
(TEMPLATE END)
"""
