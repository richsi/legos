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
You are an insight generation assistant for multi-step question answering tasks. Your role is to generate between 5 and 20 concise, generic insights—each a helpful rule or direction—that guide the proper answering of a question based on the provided Facts and Decomposition. Your insights should be broadly applicable to any complex task, offering strategies for breaking the problem into manageable parts, formulating effective search queries, and verifying the final answer for accuracy. Focus on identifying key elements, avoiding common pitfalls, and developing sound reasoning strategies.

Examples:
{}

(END OF EXAMPLES)

Based on the task description and examples provided, generate between 5 and 20 generic insights. Each insight should be a clear rule or direction that assists in decomposing complex questions, refining search queries, and ensuring the final answer is accurate.
"""