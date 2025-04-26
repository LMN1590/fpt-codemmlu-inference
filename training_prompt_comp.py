from langchain.prompts.chat import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)



ANSWER_SYSTEM = '''You are a highly skilled software engineer. Analyze the following multiple-choice programming question and select the most accurate answer

Instructions:
{core_instructions}

Follow these formatting instructions carefully:
1. Read the given multiple choice question carefully and initialize the step counter between <count> and </count> to {budget}.
2. Generate a detailed, logical step-by-step solution. There are two types of steps you may use:
2.1. A reasoning step based on analyzing the question or evaluating answer choices. Enclose your logical conclusion in <reasoning> and </reasoning> tags.
2.2. A factual step that introduces general knowledge or principles relevant to solving the problem. Enclose these facts in <fact> and </fact> tags.
3. You may use up to {budget} steps in total. Track your remaining steps by decrementing the counter within <count> and </count> tags. Do not exceed this budget, and you do not have to use all the steps.
4. When uncertain or at a decision point, pause to reflect. Engage in self-reflection about your reasoning so far and decide whether to revise a previous step or continue. Enclose your reflection in <reflection> and </reflection> tags.
5. Once your step-by-step reasoning is complete, synthesize everything into your final choice and enclose it within <answer> and </answer> tags.
6. Provide a critical, honest, and subjective evaluation of your reasoning process within <reflection> and </reflection> tags.'''

ANSWER_USER = """{question}
{choices}

Supporting Context(if any):
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(ANSWER_SYSTEM),
    HumanMessagePromptTemplate.from_template(ANSWER_USER)
])



ANSWER_REVIEW_SYSTEM = """You are a skilled software engineer. Below is a multiple-choice programming question, the correct answer, and a candidate's attempted answer with reasoning.

Your task is to:
- Acknowledge any correct aspects of the reasoning.
- Identify key errors or misconceptions compared to the ground truth.
- If incorrect or suboptimal, briefly explain why.

Keep your response concise, balanced, and focused—highlight what was done well, point out flaws in logic or understanding, and stress the importance of accuracy in programming."""
ANSWER_REVIEW_USER = """{question}
{choices}
Groundtruth: {groundtruth}
Attempted Answer:
{answer}"""
ANSWER_REVIEW_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(ANSWER_REVIEW_SYSTEM),
    HumanMessagePromptTemplate.from_template(ANSWER_REVIEW_USER)
])

IMPROVEMENT_GENERATION_SYSTEM = '''You are a skilled software engineer. Given a review of a generated answer, the current instruction text, and a list of existing improvement directions, suggest improvements to the instruction section.

Output your results as a JSON object with this key:
- "instruction_improvements": A list of focused, actionable suggestions for improving the instructions. Each suggestion should be singular, technically relevant, and written like practical programming guidance. Limit the list to {lim_direction} suggestions.

Note:
- Focus on specific programming concepts, data structures, algorithms, and language features that are commonly tested in MCQs
- Provide concrete examples of what to check for (e.g., "Verify loop boundary conditions in nested iterations")
- Address multiple programming paradigms (OOP, functional, procedural) and languages (Java, Python, C++, JavaScript, etc.)
- Include language-specific considerations where relevant but maintain broad applicability
- Include networking, database, system design, and other software engineering concepts
- Output only the JSON object. No extra text or headers.
- Remember that access to the correct answer is not given during testing.

Sample:
{{
  "instruction_improvements": [
    "Check boundary conditions in array/list operations to prevent index out-of-range errors (e.g., arr[length-1] vs arr[length])",
    "Verify SQL join conditions for potential unintended Cartesian products in multi-table queries",
    "Evaluate time complexity differences between hash-based lookups (O(1)) and binary search approaches (O(log n))",
    "Analyze closure variable scope in JavaScript functions to identify potential reference issues",
    "Test synchronization mechanisms in concurrent code for potential deadlock conditions"
  ]
}}'''

IMPROVEMENT_GENERATION_USER = """Answer Review: 
{review}

Current Instruction:
{cur_instructions}

Existing Improvement Directions: {cur_improv}"""
IMPROVEMENT_GENERATION_ASSISTANT = "{{"

IMPROVEMENT_GENERATION_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(IMPROVEMENT_GENERATION_SYSTEM),
    HumanMessagePromptTemplate.from_template(IMPROVEMENT_GENERATION_USER),
    AIMessagePromptTemplate.from_template(IMPROVEMENT_GENERATION_ASSISTANT)
])

IMPROVEMENT_APPLY_SYSTEM = '''As a prompt engineering expert, incorporate the specified improvement direction into the current iteration of the instruction set. Preserve all essential information and context. Return only the improved prompt with no explanations. 

Note:
- When faced with trade-offs, implement improvements partially rather than omitting them entirely according to your judgement.
- Don't explain yourself.
- The initial instruction set can be empty. If so, just incorporate the improvement direction as normal.
- Make your improvement in a short and concise manner, only including the core aspect of it.
- Focus on the technical details if the improvement direction mention it.
- Use the bulletin format.'''
IMPROVEMENT_APPLY_USER = """Current Instruction: {cur_instruction}
Improvement Direction: {improv_dir}"""
IMPROVEMENT_APPLY_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(IMPROVEMENT_APPLY_SYSTEM),
    HumanMessagePromptTemplate.from_template(IMPROVEMENT_APPLY_USER)
])

EVAL_SYSTEM = '''You are a highly skilled software engineer. Analyze the following multiple-choice programming question carefully. Consider the code logic, language syntax, and best practices. Select the most accurate and efficient answer.

Instructions:
{core_instructions}

Output your reasoning process and answer using the following XML-style tagged format. Do not include any additional commentary or formatting:
- <reasoning>: The concise reasoning path behind your decision, incorporating relevant real-world knowledge and coding experience. Do not include any endline characters. Keep it short and simple.
- <result>: The answer to the multiple-choice programming question. Output only the corresponding letter of the correct answer. Do not respond with "None of the above", "N/A", or "None". Your answer must match one of the provided options.
- <confidence>: Your confidence level in the answer, expressed as a float between 0 and 1.

Note: 
- Only outputs the tagged content. Do not generate any header or footer, only the tagged content.
- When providing your answer, you MUST properly replace all special XML characters within the tagged content with their text counter part. Do NOT attempt to modify the tags, only the content inside of them should be modified.
+ Replace "&lt;" with "lesser" (for < symbol)
+ Replace "&gt;" with "greater" (for > symbol)
+ Replace "&lt;=" with "lesser_than" (for <= symbol)
+ Replace "&gt;=" with "greater_than" (for >= symbol)
+ Replace "&amp;" with "and" (for & symbol)

Sample:
<reasoning>The variable a is lesser_than b.</reasoning>
<result>A</result>
<confidence>0.95</confidence>'''

EVAL_ASSISTANT = '{{'

EVAL_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(EVAL_SYSTEM),
    HumanMessagePromptTemplate.from_template(ANSWER_USER),
    AIMessagePromptTemplate.from_template(EVAL_ASSISTANT)
])