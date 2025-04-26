import json
from tqdm import tqdm
import string
from utils import check_key

with open('actual_test_extra_basic_knn_cot_ensemble5_finetuned_0.7203647.json') as file:
    qna_dict = json.load(file)

PERM = 5

count = 0
correct_ans = 0
final_res = []  
for task_id, samples in qna_dict.items():
    sample_score = [0]*26
    for sample in samples[:PERM]:
        if not check_key(sample,'answer') or not sample['answer']['status'] or sample['answer']['result'] is None or not sample['answer']['result'] in string.ascii_uppercase or string.ascii_uppercase.index(sample['answer']['result']) >= len(sample['choices']):continue
        try:
            current_index = string.ascii_uppercase.index(sample['answer']['result'])
            right_index = sample['permutation'][current_index]
            sample_score[right_index] += float(sample['answer']['confidence'])
        except Exception as e:
            # sample_score[0] += 1
            print("error:",e)
            print(sample['answer']['confidence'])
    
    max_index = sample_score.index(max(sample_score))
    count+=1
    if string.ascii_uppercase[max_index] == samples[0]['groundtruth']: correct_ans+=1

    
    
print(correct_ans)
print(count)
print(correct_ans/count)