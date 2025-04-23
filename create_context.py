import pandas as pd
import json
import string
import ast
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from typing import List
from setup import MultipleChoiceQuestion

with open('actual_training.json') as file:
    training_context:List[MultipleChoiceQuestion] = json.load(file) 

TEMPLATE = '''Question: {question}
Choices: 
{choices}

Answer: {answer}
'''

context_lst = []
for ctx in training_context:
    question = ctx['question']
    parsed_choices = ctx['choices']
    answer = ctx['groundtruth']
    if answer not in string.ascii_uppercase or string.ascii_uppercase.index(answer)>=len(parsed_choices): continue
    try:
        sample_text = TEMPLATE.format(
            question = question,
            choices = "\n".join([f'{string.ascii_uppercase[idx]}. {choice}' for idx,choice in enumerate(parsed_choices)]),
            answer = f'{answer}. {parsed_choices[string.ascii_uppercase.index(answer)]}'
        )
    except Exception as e:
        print(parsed_choices)
        print(answer)
        raise e
    context_lst.append(
        {
            "question":question,
            "sample_text":sample_text
        }
    )
print(len(context_lst))
    

QUANT_O3_MODEL = SentenceTransformer(
    "all-MiniLM-L6-v2",
    backend="onnx",
    model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"},
)
embeddings=QUANT_O3_MODEL.encode(sentences=[ques['question'] for ques in context_lst],show_progress_bar=True,normalize_embeddings=True,batch_size=32)

qdrant_points = [PointStruct(
    id = str(uuid.uuid4()),
    vector = embed,
    payload = {
        'sample_text':ques['sample_text'],
        'question':ques['question']
    }
    
) for ques,embed in zip(context_lst,embeddings)]

batches = [
    qdrant_points[i:i+2000] 
    for i in range(0,len(qdrant_points),2000)
]

client = QdrantClient(
    host = 'localhost',
    port=6333
)

for batch in tqdm(batches,desc = "Uploading to Qdrant..."):
    operation_info = client.upsert(
        collection_name = 'all-mini-lm-update-yeah',
        wait=True,
        points = batch,
    )