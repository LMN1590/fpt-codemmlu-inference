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


training_samples = {}
for i in range(4):
    FILEPATH = f'training_samples_{i}.json'
    with open(FILEPATH) as file:
        ques_dict:List[MultipleChoiceQuestion] = json.load(file)
    for item in ques_dict:
        training_samples[item['label']] = training_samples.get(item['label'],[])+[item]
    
unique_topics = {topic:lst for topic,lst in training_samples.items() if len(lst)<20}
general_topics = {topic:lst for topic,lst in training_samples.items() if len(lst)>=20}
with open('normal_training_sample.json','w') as file:
    json.dump(general_topics,file)
with open('unique_training_sample.json','w') as file:
    json.dump(unique_topics,file)