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
- Prefer using items from the existing list where applicable. Only introduce new directions if necessary.
- Keep suggestions technical but concise—aim for clarity and utility (e.g., "Remind the model to check for Python operator precedence.").
- Please be specific on your advice, not just general advice like "Be careful with your decision".
- Output only the JSON object. No extra text or headers.
- Remember that access to the correct answer is not given during testing.

Sample:
{{
  "instruction_improvements": [
    "Remind the model to check for off-by-one errors in loop boundaries.",
    "Encourage explicit mention of edge cases in reasoning."
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

Output your reasoning process and answer in the form of a JSON with three keys:
- "reasoning": The concise reasoning path behind your decision, taking into account real-world knowledges and your coding experience. Please perform your reasoning step-by-step. Do not include any endline.
- "result": The answer to the multiple-choice programming question. Output only the corresponding letter of the correct answer. Do not answer with "None of the above", "N/A" or "None". Your choice must be among the provided answer.
- "confidence": How confidence you are in your answer.

Note: 
- Only outputs the JSON. Do not generate any header or footer, only the JSON.
- Do not include any endline in the JSON response.

Sample:
{{
    "reasoning": "...",
    "result": "A"|"B"|"C"|"D"|"E"|"F"...,
    "confidence": [0-1]
}}'''

EVAL_ASSISTANT = '{{'

EVAL_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(EVAL_SYSTEM),
    HumanMessagePromptTemplate.from_template(ANSWER_USER),
    AIMessagePromptTemplate.from_template(EVAL_ASSISTANT)
])