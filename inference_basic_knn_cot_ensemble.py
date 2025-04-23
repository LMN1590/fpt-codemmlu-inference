import string
from typing import Dict,List
import json
import traceback
import time
from tqdm import tqdm
from setup import MultipleChoiceQuestion,Result,llm_together
from utils import check_key
from langchain_core.runnables import RunnableSerializable

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

def inference(sample:MultipleChoiceQuestion,answer_pipeline:RunnableSerializable):
    question = sample['question']
    choices = sample['choices']
    context = sample['context']

    result = answer_pipeline.invoke({
        "question":question,
        "choices": "\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(choices)],),
        'context':'\n'.join(context)
    })
    
    result_content = result.content if result.content[0]=='{' else '{' + result.content
    result_content = result_content if result.content[-1]=='}' else result_content + '}'

    try:
        parsed_result = parse_json(result_content)
        return Result(
            reasoning = parsed_result['reasoning'] if check_key(parsed_result,'reasoning') else 'No Reasoning Found',
            result = parsed_result['result'],
            confidence=parsed_result['confidence'] if check_key(parsed_result,'confidence') else 'No Confidence Found',
            raw_response=result.content,
            status=True,
            error_messages=''
        )
    except Exception as e:
        return Result(
            reasoning = result.content,
            result = 'Empty',
            confidence=0,
            raw_response=result.content,
            status=False,
            error_messages=f'Error occurs. {traceback.format_exc()}'
        )

PATIENCE = 5
SPLIT_SIZE = 320
WAIT_TIME = 15
PERM_SIZE = 5

from prompt import COT_PROMPT_PROMPT
FILEPATH = f'actual_test_extra_basic_knn_cot_ensemble5_pro.json'
if __name__ == '__main__':
    with open(FILEPATH) as file:
        ques_dict:Dict[str,List[MultipleChoiceQuestion]] = json.load(file)
    rate_limited_no_more = False
    for sample in tqdm([
        sample 
        for samples in list(ques_dict.values())
        for sample in samples[:PERM_SIZE]
        if (
            not check_key(sample,'answer') 
            or not sample['answer']['status']
            or sample['answer']['result'] is None
            or not sample['answer']['result'] in string.ascii_uppercase
            or string.ascii_uppercase.index(sample['answer']['result']) >= len(sample['choices'])
        )]):
        for i in range(PATIENCE):
            try:
                sample['answer'] = inference(sample,COT_PROMPT_PROMPT|llm_together)
                break
            except Exception as e:
                print('Rate limit for gemini achieved, resting for 5 secs')
                time.sleep(WAIT_TIME)
                if(i==PATIENCE-1): rate_limited_no_more = True
        if rate_limited_no_more: 
            print('Not Done Yet :((((')
            break
            

    with open(FILEPATH,'w') as file:
        json.dump(ques_dict,file)