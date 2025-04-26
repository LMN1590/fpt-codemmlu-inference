import string
from typing import Dict,List
import json
import traceback
import time
from tqdm import tqdm
from setup import MultipleChoiceQuestion,Result,llm_together
from utils import check_key
from langchain_core.runnables import RunnableSerializable
import xml.etree.ElementTree as ET 
from bs4 import BeautifulSoup
import random

def parse_xml(llm_output:str):
    soup = BeautifulSoup(llm_output, "html.parser")
    output_res = {
        'reasoning':soup.find("reasoning").text,
        'result':soup.find("result").text,
        'confidence':soup.find("confidence").text
    }
    return output_res


def inference(sample:MultipleChoiceQuestion,answer_pipeline:RunnableSerializable):
    question = sample['question']
    choices = sample['choices']
    context = sample['context']
    professional_fields = sample['professionals']

    result = answer_pipeline.invoke({
        'professional_field':random.choice(professional_fields),
        "question":question,
        "choices": "\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(choices)],),
        'context':'\n'.join(context)
    })
    # print(result.content)
    try:
        parsed_result = parse_xml(result.content)
        return Result(
            reasoning = parsed_result['reasoning'] if check_key(parsed_result,'reasoning') else 'No Reasoning Found',
            result = parsed_result['result'],
            confidence=parsed_result['confidence'] if check_key(parsed_result,'confidence') else 'No Confidence Found',
            raw_response=result.content,
            status=True,
            error_messages=''
        )
    except Exception as e:
        print(result.content)
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
WAIT_TIME = 5
PERM_SIZE = 1

from prompt import EXTRA_PRO_VIP_PROMPT
FILEPATH = f'actual_test_extra_automed.json'
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
                sample['answer'] = inference(sample,EXTRA_PRO_VIP_PROMPT|llm_together)
                with open(FILEPATH,'w') as file:
                    json.dump(ques_dict,file)
                break
            except Exception as e:
                print('Rate limit for gemini achieved, resting for 5 secs')
                time.sleep(WAIT_TIME)
                if(i==PATIENCE-1): rate_limited_no_more = True
        if rate_limited_no_more: 
            print('Not Done Yet :((((')
            break
            

    