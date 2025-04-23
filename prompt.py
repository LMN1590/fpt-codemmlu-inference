from langchain.prompts.chat import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)

GENERATE_LABEL_SYSTEM = '''You are a highly skilled software engineer. Your task is to analyze a multiple-choice programming question and determine the most appropriate topic label that captures the core concept being tested.

Instructions:
- Carefully read the programming question and its choices.
- Identify the main concept or knowledge area the question is testing.
- Choose a topic label that is:
    + General enough to group similar questions (e.g., "C Programming" instead of "C Enums"),
    + But still informative and focused enough to reflect what kind of knowledge the question targets.
- Use an existing label from the provided list if it fits.
- If none of the provided labels match the question's concept accurately, create a new, appropriate topic label.
- Prefer more general topic labels when they already encompass the concept. For example:
    + Use "C Programming" instead of "C Enums" or "C Sizeof Operator" if the question tests common behavior within C.
    + Use "Python Basics" instead of "Python List Indexing" for typical syntax or basic operations.
- Avoid labels that are too specific to the example or question phrasing.
- Output your result as a JSON object with one key: "topic_label" and a list of topic label(s) as its value.
- Include {min_gen_label} topic labels, prioritize the most general one if applicable.
- Do not include any explanation, only the JSON.

Sample:
{{
    "topic_label": ["Linux Kernel Command"]
}}'''

GENERATE_LABEL_USER = '''Question: {question}
Choices: 
{choices}

Answer: {answer}

Provided List of existing topic label: {existing_list}'''

GENERATE_LABEL_ASSISTANT = '{{'

GENERATE_LABEL_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(GENERATE_LABEL_SYSTEM),
    HumanMessagePromptTemplate.from_template(GENERATE_LABEL_USER),
    AIMessagePromptTemplate.from_template(GENERATE_LABEL_ASSISTANT)
])


BASIC_PROMPT_SYSTEM = """You are a highly skilled software engineer. Analyze the following multiple-choice programming question and select the correct answer.

Output your reasoning process and answer in the form of a JSON with three keys:  
- "result": The answer to the multiple-choice programming question. Output only the corresponding letter of the correct answer. Do not answer with "None of the above" or "N/A". Your choice must be among the provided answer.

Note: Only outputs the JSON. Do not generate any header or footer, only the JSON.

Sample:
{{
    "result": "A"|"B"|"C"|"D"|"E"|"F"...
}}
"""
BASIC_PROMPT_USER = """{question}
{choices}

Supporting Context(if any):
{context}"""
BASIC_PROMPT_ASSISTANT = "{{"
BASIC_PROMPT_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(BASIC_PROMPT_SYSTEM),
    HumanMessagePromptTemplate.from_template(BASIC_PROMPT_USER),
    AIMessagePromptTemplate.from_template(BASIC_PROMPT_ASSISTANT)
])



COT_PROMPT_SYSTEM = """You are a highly skilled software engineer. Analyze the following multiple-choice programming question carefully. Consider the code logic, language syntax, and best practices. Select the most accurate and efficient answer.

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
}}
"""
COT_PRO_PROMPT_SYSTEM = """You are a highly skilled professional in the field of {professional_field}. Analyze the following multiple-choice programming question carefully. Consider the code logic, language syntax, and best practices. Select the most accurate and efficient answer.

Output your reasoning process and answer in the form of a JSON with three keys:
- "reasoning": The concise reasoning path behind your decision, taking into account real-world knowledges and your coding experience. Please perform your reasoning step-by-step. Do not include any endline.
- "result": The answer to the multiple-choice programming question. Output only the corresponding letter of the correct answer. Do not answer with "None of the above", "N/A" or "None". If you can't come up with an answer, choose the most likely answer.
- "confidence": How confidence you are in your answer.

Note: 
- Only outputs the JSON. Do not generate any header or footer, only the JSON.
- Do not include any endline in the JSON response.

Sample:
{{
    "reasoning": "...",
    "result": "A"|"B"|"C"|"D"|"E"|"F"...,
    "confidence": [0-1]
}}
"""
COT_PROMPT_USER = """{question}
{choices}

Supporting Context(if any):
{context}"""
COT_PROMPT_ASSISTANT = "{{"
COT_PROMPT_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(COT_PROMPT_SYSTEM),
    HumanMessagePromptTemplate.from_template(COT_PROMPT_USER),
    AIMessagePromptTemplate.from_template(COT_PROMPT_ASSISTANT)
])
COT_PRO_PROMPT_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(COT_PRO_PROMPT_SYSTEM),
    HumanMessagePromptTemplate.from_template(COT_PROMPT_USER),
    AIMessagePromptTemplate.from_template(COT_PROMPT_ASSISTANT)
])


GET_TEST_LABEL_SYSTEM = '''You are a highly skilled software engineer. Your task is to analyze a multiple-choice programming question and determine the most appropriate topic label that captures the core concept being tested.

Instructions:
- Carefully read the programming question and its choices.
- Identify the main concept or knowledge area the question is testing.
- Choose a topic label representing the underlying concepts or knowledge being asked.
- Prefer more general topic labels when they already encompass the concept. For example:
    + Use "C Programming" instead of "C Enums" or "C Sizeof Operator" if the question tests common behavior within C.
    + Use "Python Basics" instead of "Python List Indexing" for typical syntax or basic operations.
- Avoid labels that are too specific to the example or question phrasing.
- Output your result as a JSON object with one key: "topic_label" and a list of topic label(s) as its value. Ensure that all the labels do not overlap with each other.
- Include {min_gen_label} topic labels, prioritize the most general one if applicable.
- Do not include any explanation, only the JSON.

Sample:
{{
    "topic_label": ["Linux Kernel Command"]
}}'''

GET_TEST_LABEL_USER = '''{question}
Choices: 
{choices}

Answer: {answer}'''

GET_TEST_LABEL_ASSISTANT = '{{'

GET_TEST_LABEL_PROMPT = ChatPromptTemplate.from_messages(messages=[
    SystemMessagePromptTemplate.from_template(GET_TEST_LABEL_SYSTEM),
    HumanMessagePromptTemplate.from_template(GET_TEST_LABEL_USER),
    AIMessagePromptTemplate.from_template(GET_TEST_LABEL_ASSISTANT)
])
