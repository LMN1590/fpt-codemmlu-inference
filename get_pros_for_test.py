from typing import List,Dict
from setup import MultipleChoiceQuestion
from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    Filter,FieldCondition,QueryRequest,
    MatchValue,MatchAny
)
from qdrant_client import QdrantClient
import random
import string
import json
from tqdm import tqdm
import time

from prompt import GET_TEST_LABEL_PROMPT
from setup import llm_together
from utils import check_key

label_pipeline =GET_TEST_LABEL_PROMPT|llm_together

def parse_json(llm_output:str):
    text = llm_output.strip()
    start = min((i for i in (text.find('{'), text.find('[')) if i != -1), default=-1)
    if start == -1: raise ValueError("No JSON found.")

    stack, in_str, esc = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        esc = (c == '\\' and in_str and not esc or False)
        in_str = in_str ^ (c == '"' and not esc) if c == '"' or in_str else in_str
        if in_str: continue
        if c in '{[': stack.append(c)
        elif c in '}]':
            if not stack or {'{': '}', '[': ']'}[stack.pop()] != c:
                raise ValueError("Mismatched brackets.")
            if not stack:
                return json.loads(text[start:i+1])
    raise ValueError("Incomplete JSON block.")

def get_pros_field(item:MultipleChoiceQuestion)->List[str]:
    question = item['question']
    choices = item['choices']
    answer = item['groundtruth']
    
    response = label_pipeline.invoke({
        'min_gen_label':10,
        'question':question,
        'choices':"\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(choices)]),
        'answer':f'{answer}. {choices[string.ascii_uppercase.index(answer)]}'
    })
    result_content = response.content if response.content[0]=='{' else '{' + response.content
    result_content = result_content if response.content[-1]=='}' else result_content + '}'
    parsed_result = parse_json(result_content)
    return parsed_result['topic_label']

PATIENCE = 5
SPLIT_SIZE = 320
WAIT_TIME = 15

if __name__ == '__main__':
    with open('actual_test_extra.json') as file:
        ques_dict:Dict[str,List[MultipleChoiceQuestion]] = json.load(file)
    rate_limited_no_more = False
    for sample in tqdm([
        sample 
        for samples in list(ques_dict.values())
        for sample in samples[:1]
        if (
            not check_key(sample,'labels') 
            or not sample['labels']
        )]):
        for i in range(PATIENCE):
            try:
                sample['labels'] = get_pros_field(sample)
                break
            except Exception as e:
                print('Rate limit for gemini achieved, resting for 5 secs')
                time.sleep(WAIT_TIME)
                if(i==PATIENCE-1): rate_limited_no_more = True
        if rate_limited_no_more: 
            print('Not Done Yet :((((')
            break
            

    with open('actual_test_extra.json','w') as file:
        json.dump(ques_dict,file)