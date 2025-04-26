import json
import dirtyjson
from typing import List,Dict
import random
import string
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET 

from langchain_core.runnables import RunnableSerializable
from langchain_core.messages import BaseMessage

from setup import MultipleChoiceQuestion,llm_together
from training_prompt_comp import (
    ANSWER_PROMPT,
    
    ANSWER_REVIEW_PROMPT,
    IMPROVEMENT_GENERATION_PROMPT,
    IMPROVEMENT_APPLY_PROMPT,
    
    EVAL_PROMPT
)


with open('actual_training.json') as file:
    training_set:List[MultipleChoiceQuestion] = json.load(file)
with open('actual_val.json') as file:
    val_set:List[MultipleChoiceQuestion] = json.load(file)
    
SAMPLE_SIZE = 32
EPOCH = 5
EARLY_STOPPING_BATCH_PATIENCE = 5
BUDGET = 5
LIM_DIRECTION = 3
RANDOM_ORDER =len(val_set)

# CORE_INSTRUCTIONS_COMP = '''Review the Multiple-Choice Question and its answer carefully before making your decision'''
CORE_INSTRUCTIONS_COMP='''- Review the Multiple-Choice Question and its answer carefully.'''

answer_pipeline = ANSWER_PROMPT|llm_together
review_pipeline = ANSWER_REVIEW_PROMPT|llm_together
gen_improv_pipeline = IMPROVEMENT_GENERATION_PROMPT|llm_together
prompt_tuning = IMPROVEMENT_APPLY_PROMPT|llm_together
eval_pipeline = EVAL_PROMPT|llm_together

def parse_LLM_output_to_valid_JSON(output:str)->str:
    return output.replace("\n"," ").replace("\t"," ")
def parse_json(llm_output:str):
    ls = re.findall('{[^{}]+}','{'+llm_output+'}')
    if(len(ls)==0): raise NotImplementedError('No JSON found')
    return dirtyjson.loads(parse_LLM_output_to_valid_JSON(ls[0]))
def parse_xml(llm_output:str):
    tree = ET.fromstring('<xml>'+llm_output+'</xml>')
    output_res = {}
    for child in tree:
        output_res[child.tag] = child.text
    return output_res

# region Utils Inference Function
def _inference(ques:MultipleChoiceQuestion,pipeline:RunnableSerializable[Dict,BaseMessage],core_instructions_comp:str):
    question = ques['question']
    choices = ques['choices']
    
    result = pipeline.invoke({
        'core_instructions':core_instructions_comp,
        'budget':BUDGET,
        "question":question,
        "choices": "\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(choices)],),
        'context':'No Context'
    })
    return result.content
def _cross_check_answer(ques:MultipleChoiceQuestion,inference_answer:str):
    question = ques['question']
    choices = ques['choices']
    groundtruth = ques['groundtruth']
    
    result = review_pipeline.invoke({
        "question":question,
        "choices": "\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(choices)],),
        'groundtruth':f'{groundtruth}. {choices[string.ascii_uppercase.index(groundtruth)]}',
        'answer':inference_answer
    })
    return result.content
def _gen_improvement_points(review:str,review_dict:Dict[str,int]):
    result = gen_improv_pipeline.invoke({
        'lim_direction':LIM_DIRECTION,
        'review':review,
        'cur_instructions':CORE_INSTRUCTIONS_COMP,
        'cur_improv':list(review_dict.keys())
    })
    
    # result_content = result.content if result.content[0]=='{' else '{' + result.content
    # result_content = result_content if result.content[-1]=='}' else result_content + '}'
    try:
        parsed_result = parse_json(result.content)
    except Exception as e:
        parsed_result = {"instruction_improvements":[]}
        print("Error Occurs", e)
        print(result.content)
    for point in parsed_result['instruction_improvements']:
        review_dict[point] = review_dict.get(point,0)+1
    return review_dict
# endregion
    
def _single_inference_and_review(ques:MultipleChoiceQuestion,pipeline:RunnableSerializable,review_dict:Dict[str,int])->Dict[str,int]:
    inference_result_content = _inference(ques,pipeline,CORE_INSTRUCTIONS_COMP)
    # with open('inference.txt','w') as file:
    #     print(inference_result_content,file=file)
    review = _cross_check_answer(ques,inference_result_content)
    # with open('review.txt','w') as file:
    #     print(review,file=file)
    review_dict = _gen_improvement_points(review,review_dict)
    # with open('review_dict.txt','w')as file:
    #     print(review_dict,file=file)
    # print(review_dict)
    # raise NotImplementedError()
    return review_dict
    

def optimize_batch(batch:List[MultipleChoiceQuestion],epoch:int,batch_idx:int,batch_length:int)->Dict[str,int]:
    current_review_dict:Dict[str,int] = {}
    for ques in tqdm(batch,desc=f'Running Inference for Epoch {epoch} - Batch {batch_idx+1}/{batch_length}'):
        current_review_dict = _single_inference_and_review(ques,answer_pipeline,current_review_dict)
    return current_review_dict
def modify_prompt(improvement_direction:str):
    print(improvement_direction)
    result = prompt_tuning.invoke({
        "cur_instruction":CORE_INSTRUCTIONS_COMP,
        'improv_dir':improvement_direction
    })
    return result.content
def run_eval(current_prompt:str,epoch:int,batch_idx:int,batch_length:int)->float:
    correct_count = 0
    count = 1e-5
    for ques in tqdm(random.sample(val_set,k=RANDOM_ORDER),desc=f'Running Eval for Epoch {epoch} - Batch {batch_idx+1}/{batch_length} - Result: {correct_count/count}'):
        result_content = _inference(ques,eval_pipeline,current_prompt)
        # result_content = result_content if result_content[0]=='{' else '{' + result_content
        # result_content = result_content if result_content[-1]=='}' else result_content + '}'
        try:
            parsed_result = parse_xml(result_content)
            correct_count += int(ques['groundtruth']==parsed_result['result'])
            count+=1
        except Exception as e:
            print(f"Failed Evaluation of ques {ques['task_id']} due to",e)
            print(result_content)
    return correct_count/RANDOM_ORDER

eval_result_record = [0]
current_patience = 0
early_stopping = False
for epoch in range(EPOCH):
    shuffled_training_set = random.sample(training_set,k=len(training_set))
    training_batches = [shuffled_training_set[i:i+SAMPLE_SIZE] for i in range(0,len(shuffled_training_set),SAMPLE_SIZE)]
    batch_length = len(training_batches)
    
    for idx,batch in enumerate(training_batches):
        review_dict = optimize_batch(batch,epoch,idx,batch_length)
        best_improvment_dir = "\n".join(list(review_dict.keys())) #max(*list(review_dict.items()),key=lambda k_v:k_v[1])
        modified_instruction_prompt = modify_prompt(best_improvment_dir)
        eval_res = run_eval(modified_instruction_prompt,epoch,idx,batch_length)
        best_eval_res = max(eval_result_record)
        if eval_res > best_eval_res:
            CORE_INSTRUCTIONS_COMP = modified_instruction_prompt
            current_patience = 0
            print(f'Updated Core Instructions: {eval_res}')
        else:
            current_patience += 1
            print(f'Evaluation Result Failed: {eval_res}')
        eval_result_record.append(eval_res)
        print({
            'current_instructions': CORE_INSTRUCTIONS_COMP,
            "eval_result": eval_result_record,
            "review_dict": review_dict,
            "best_improvement_dir": best_improvment_dir,
            'updated_prompt': modified_instruction_prompt,
            "current_patience":current_patience
        })

        print('---------------')
        if current_patience == EARLY_STOPPING_BATCH_PATIENCE: 
            early_stopping = True
            break
    if early_stopping: break
        
with open('finetuned_instruction_prompt.txt','w') as file:
    print(CORE_INSTRUCTIONS_COMP,file=file)
        