from typing import List,Union
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langchain.prompts.chat import(
    ChatPromptTemplate,ChatPromptValue
)
import uuid

from dotenv import load_dotenv
_ = load_dotenv('./.env')

class Result(TypedDict):
    reasoning:str
    result:str
    status:bool
    confidence:float
    raw_response:str
    error_messages:str

class MultipleChoiceQuestion(TypedDict):
    task_id:str
    
    question:str
    choices:List[str]
    context:List[str]
    permutation:List[int]
    
    prompt:ChatPromptValue
    professionals:List[str]
    
    answer:Result
    groundtruth:str
    label:str

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )
llm_together = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0.5,
    max_tokens=1024,
    timeout=None,
    max_retries=2,api_key='tgp_v1_x77ByU1OKQEtMf3NjWTXfnh6ABXwthcrdOW3n13h1fE'
)