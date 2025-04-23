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

from prompt import GET_TEST_LABEL_PROMPT
from setup import llm_together

QUANT_O3_MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2",
    backend="onnx",
    model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"},
)
CLIENT = QdrantClient(
    host = 'localhost',
    port=6333
)

PERMUTATION_COUNT = 10
RETRIEVAL_CONTEXT = 3
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


def generate_context_and_perm(question_set:List[MultipleChoiceQuestion])->Dict[str,List[MultipleChoiceQuestion]]:
    ques_dict = {}
    
    ques_embeddings = QUANT_O3_MODEL.encode([ques['question'] for ques in question_set],normalize_embeddings=True,batch_size=32,show_progress_bar=True)
    key_embed_queries = [
        QueryRequest(
            query = embedding,
            limit = 3, 
            with_payload=True,with_vector=False
        )
        for embedding in ques_embeddings
    ]
    query_responses = CLIENT.query_batch_points(
        collection_name='all-mini-lm-update-yeah',
        requests=key_embed_queries
    )
    
    for ques,resp in tqdm(zip(question_set,query_responses)):
        choices_len = len(ques['choices'])
        
        labels = get_pros_field(ques) 
        
        ques_dict[ques['task_id']] = [
            {
                **ques,
                'context': [
                    point.payload['sample_text']
                    for point in resp.points
                ],
                'pros_labels': labels
            }
        ]
        perm_set = set()
        perm_set.add(str(list(range(choices_len))))
        
        cur_perm_count = min(PERMUTATION_COUNT,choices_len*(choices_len-1))
        
        while len(perm_set)<cur_perm_count:
            # print(len(perm_set))
            permutation = random.sample(list(range(choices_len)), choices_len)
            if str(permutation) in perm_set: continue
            ques_item = {
                **ques,
                "choices":[f'{ques["choices"][real_idx]}' for idx,real_idx in enumerate(permutation)],
                'permutation': permutation,
                'groundtruth': string.ascii_uppercase[permutation.index(string.ascii_uppercase.index(ques['groundtruth']))],
                'context': [
                    point.payload['sample_text']
                    for point in resp.points
                ],
                'pros_labels': labels
            }
            perm_set.add(str(permutation))
            ques_dict[ques['task_id']].append(ques_item)
        
    return ques_dict


if __name__=='__main__':
    with open('actual_test.json') as file:
        ques_lst = json.load(file)
    ques_dict = generate_context_and_perm(ques_lst)
    with open('actual_test_extra.json','w')as file:
        json.dump(ques_dict,file)        