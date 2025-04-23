import json
from tqdm import tqdm
import string
from utils import check_key

with open('actual_test_extra_basic_knn_cot.json') as file:
    ques_dict = json.load(file)

count = 0
correct_ans = 0
for sample in tqdm([
    sample 
    for samples in list(ques_dict.values())
    for sample in samples[:1]
    if not (
        not check_key(sample,'answer') 
        or not sample['answer']['status']
        or sample['answer']['result'] is None
        or not sample['answer']['result'] in string.ascii_uppercase
        or string.ascii_uppercase.index(sample['answer']['result']) >= len(sample['choices'])
    )]):
    count+=1
    if sample['answer']['result'] == sample['groundtruth']: correct_ans+=1
print(correct_ans)
print(count)
print(correct_ans/count)