import json
import random
with open('normal_training_sample.json') as file:
    general_sample = json.load(file)
    
with open('unique_training_sample.json') as file:
    unique_sample = json.load(file)

samples = [[],[],[]]

TRAIN_VAL_TEST_PERCENT = [10,5,5]
for topic,lst in general_sample.items():
    topic_length = len(lst)
    counts = [int(percent*topic_length/100) for percent in TRAIN_VAL_TEST_PERCENT]
    random_idx = random.sample(lst,k=sum(counts))
    
    start_count = 0
    for count_idx,count in enumerate(counts):
        topic_sample = [done_sample for done_sample in random_idx[start_count:start_count+count]]
        samples[count_idx] += topic_sample
        start_count += count
    print([len(sample) for sample in samples])
    
samples[2] += [item for topic,lst in unique_sample.items() for item in lst]
print([len(sample) for sample in samples])

with open('actual_training.json','w') as file:
    json.dump(samples[0],file)
with open('actual_val.json','w') as file:
    json.dump(samples[1],file)
with open('actual_test.json','w') as file:
    json.dump(samples[2],file)