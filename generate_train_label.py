import pandas as pd
import string
import ast
import json
from typing import List
from tqdm import tqdm
import time

from prompt import GENERATE_LABEL_PROMPT
from setup import llm_together,MultipleChoiceQuestion
from utils import parse_LLM_output_to_valid_JSON,check_key

label_pipeline = GENERATE_LABEL_PROMPT|llm_together
existing_labels = set(['C Programming Language', 'Linux Kernel Command', 'SQL Commands'])

def generate_label(item):
    question = item['question']
    choices = item['choices']
    answer = item['answer']
    global existing_labels
    
    response = label_pipeline.invoke({
        'min_gen_label':1,
        'question':question,
        'choices':"\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(choices)]),
        'answer':f'{answer}. {choices[string.ascii_uppercase.index(answer)]}',
        'existing_list':list(existing_labels)
    })
    ls = response.content if response.content[0]=='{' else '{' + response.content
    print(response.content)
    parsed_result = json.loads(parse_LLM_output_to_valid_JSON(ls))
    existing_labels = existing_labels | set(parsed_result['topic_label'])
    
    
    item['label'] = parsed_result['topic_label'][0]
    # print(existing_labels)
    # print('----')
    return item


def parse_training_data():
    df = pd.read_csv('b6_train_data.csv')
    df = df.dropna().reset_index(drop=True)
    
    training_lst = []
    for ques_id,question,choices,answer in zip(df['task_id'],df['question'],df['choices'],df['answer']):
        answer = answer.replace('ANSWER: ','')
        parsed_choices = ast.literal_eval(choices)
        choices_len = len(parsed_choices)
        if answer not in string.ascii_uppercase or string.ascii_uppercase.index(answer)>=len(parsed_choices): continue
        
        training_lst.append(MultipleChoiceQuestion(
            question = question,
            choices = parsed_choices,
            answer = answer,
            permutation = list(range(choices_len))
        ))
    return training_lst


SPLIT_SIZE = 800
TRAINING_BATCH = 3
PATIENCE = 5
WAIT_TIME = 10

if __name__ == '__main__':
    # ques_dict:List[MultipleChoiceQuestion] = parse_training_data()
    # batches = [ques_dict[i:i+SPLIT_SIZE]  for i in range(0,len(ques_dict),SPLIT_SIZE)]
    # for idx,batch in enumerate(batches):
    #     with open(f'training_samples_{idx}.json','w') as file:
    #         json.dump(batch,file)
    
    FILEPATH = f'training_samples_{TRAINING_BATCH}.json'
    with open(FILEPATH) as file:
        ques_dict:List[MultipleChoiceQuestion] = json.load(file)
    
    rate_limited_no_more = False
    for idx, item in enumerate(tqdm(ques_dict,desc=f'Processing Training Batch {TRAINING_BATCH}')):
        if check_key(item,'label'): continue
        for i in range(PATIENCE):
            try:
                ques_dict[idx]= generate_label(item)
                break
            except Exception as e:
                print(f'Rate limit for gemini achieved, resting for {WAIT_TIME} secs')
                # raise e
                time.sleep(WAIT_TIME)
                if(i==PATIENCE-1): 
                    rate_limited_no_more = True
                    print(e)
        if rate_limited_no_more: 
            print('Not Done Yet :((((')
            break
    with open(FILEPATH,'w') as file:
        json.dump(ques_dict,file)