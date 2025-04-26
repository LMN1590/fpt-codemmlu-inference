import json
from tqdm import tqdm
import string
from utils import check_key

with open('actual_test_extra_automed_0.7042682.json') as file:
    ques_dict = json.load(file)
UNIQUE = ['Linux Kernel Command', 'Shell Scripting', 'Accessibility', 'Web Development', 'CI/CD Pipelines', 'Security', 'Networking', 'Audio Compression', 'JSP', 'Cloud Computing', 'Geometry', 'Distributed Computing', 'Computer Architecture', 'Operating System', 'Data Mining', 'Artificial Intelligence', 'Linear Programming', 'XML', 'CSS', 'Kotlin Programming', 'Database Management', 'Project Management', 'Tree Traversal', 'Graph Theory', 'Hash Table', 'None of the above', 'Java', 'Analogy', 'Object-Oriented Programming', 'Regular Expressions', 'HTML/CSS', 'JavaScript', 'Mathematics']
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
    ) and sample['label'] in UNIQUE
    ]):
    count+=1
    if sample['answer']['result'] == sample['groundtruth']: correct_ans+=1
print(correct_ans)
print(count)
print(correct_ans/count)