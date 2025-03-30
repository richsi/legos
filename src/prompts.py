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