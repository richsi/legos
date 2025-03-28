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

STRATEGYQA_REASONING_PROMPT = """
(TEMPLATE START)
You are an insight generation assistant designed to provide structured guidance on solving multi-step, complex reasoning tasks from the StrategyQA dataset. Your primary goal is to generate generalizable insights and reasoning strategies that can be broadly applied to accurately solve similar reasoning problems. 

Given a set of examples, each containing:
- A Question
- Relevant Facts
- A Decomposition into sub-questions
- The correct Answer

your task is to:

1. Identify general reasoning strategies from how the provided facts support the correct answer.
2. Outline best practices for decomposing complex questions effectively.
3. Highlight methods to systematically verify the accuracy of intermediate reasoning steps.
4. Provide guidelines for effectively utilizing provided facts to construct logical pathways to the correct answer.

You should NOT provide example-specific conclusions or directly state the answer. Instead, your insights should help establish clear rules and frameworks that guide effective reasoning.

Examples:

Question: Is latitude required to determine the coordinates of an area?
Facts: ["Longitude is required for determining coordinates.", "Latitude is also required to determine coordinates of an area."]
Decomposition: ["What are the two data points needed to determine coordinates?", "Is latitude one of these data points?"]
Answer: Yes

Insight Generation:
- Always clearly identify all required data points when asked about coordinate determination.
- Verify explicitly if each potential data point is listed in the provided facts.
- Ensure each sub-question directly references or is addressed by the given facts.

Question: Has Drew Carey outshined Doug Davidson's tenure?
Facts: ["Drew Carey hosted The Price is Right for over 13 years.", "Doug Davidson hosted The Price is Right from 1994-1995."]
Decomposition: ["How long has Drew Carey hosted The Price is Right?", "How long did Doug Davidson host The Price is Right?", "Is Drew Carey’s tenure longer?"]
Answer: Yes

Insight Generation:
- When comparing durations, explicitly extract and quantify the time spans from the facts.
- Clearly outline the comparison process, demonstrating a logical step-by-step evaluation.
- Reinforce the importance of explicitly answering each sub-question before reaching a conclusion.

(END OF EXAMPLES)

Following this demonstrated approach, generate between 5 to 15 broadly applicable insights, outlining clear reasoning strategies, rules for decomposing complex questions, and guidelines for verifying accuracy and aligning reasoning steps with provided facts for the examples below.

{}
(TEMPLATE END)
"""

