# A statistics of captions
import json
from bert import sent2words
from tqdm import tqdm

json_path = '/home/wuziqiang/data/CUHK-PEDES/CUHK_PEDES_prepare/reid_raw_np.json'
with open(json_path, 'r') as f:
    json_file = json.load(f) 

word_count = dict()
np_fine_count = dict()
np_coarse_count = dict()

for json_item in tqdm(json_file):
    for caption in json_item['captions']:
        words = sent2words(caption)
        for word in words:
            if word in word_count.keys():
                word_count.update({word: word_count[word]+1})
            else:
                word_count.update({word: 1})
    for fine in json_item['np_fine']:
        for np in fine:
            np = ' '.join(sent2words(np))
            if np in np_fine_count.keys():
                np_fine_count.update({np: np_fine_count[np]+1})
            else:
                np_fine_count.update({np: 1})

    for coarse in json_item['np_coarse']:
        for np in coarse:
            np = ' '.join(sent2words(np))
            if np in np_coarse_count.keys():
                np_coarse_count.update({np: np_coarse_count[np]+1})
            else:
                np_coarse_count.update({np: 1})
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
for key in np_fine_count.keys():
    if np_fine_count[key] == 1:
        count_1 += 1
    elif np_fine_count[key] == 2:
        count_2 += 1
    elif np_fine_count[key] == 3:
        count_3 += 1
    elif np_fine_count[key] == 4:
        count_4 += 1
print('count1:', count_1)
print('count2:', count_2)
print('count3:', count_3)
print('count4:', count_4)
print('Vocabulary size:{}'.format(len(word_count.keys())))
print('Fine np num:{}'.format(len(np_fine_count.keys())))
print('Coarse np num:{}'.format(len(np_coarse_count.keys())))
