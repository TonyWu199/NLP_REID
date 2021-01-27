import json
import scipy.io as sio
import h5py
import numpy as np
import os
from PIL import Image
import argparse
import re
import pdb
from bert import *
from transformers import BertTokenizer
from tqdm import tqdm

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
suffix = '.npy'
json_file = 'reid_raw.json'
sent_per_img = 2
dim = 768

print('==>initiate bert tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

bert = bert('base').to(device)

#setting
parser = argparse.ArgumentParser()
parser.add_argument('--case', default='case', type=str)#not implement
parser.add_argument('--maxlength', default=80, type=int)
opt = parser.parse_args()

case = opt.case
maxlength = opt.maxlength

#input
datasetname = 'cuhkpedes'
inputpath = '/home/wuziqiang/data/CUHK-PEDES/CUHK_PEDES_prepare'
jsonpath = os.path.join(inputpath, json_file)
imgdirpath = os.path.join(inputpath, 'imgs')

split = [0 for i in range(34054)] + [1 for i in range(3078)] + [2 for i in range(3074)]

outputdirpath = '{}_{}_maxlen={}_{}_average'.format(datasetname, case, maxlength, dim)
if not os.path.exists(outputdirpath):
    os.mkdir(outputdirpath)
with open(jsonpath, 'r') as f:
    jsondata = json.load(f)
assert len(jsondata) == len(split)
captions = []
tokens = []
imgpaths = []
labels = []

'''
    Load json data to list
'''
for sample in jsondata:
    captions.append(sample['captions'])#still unicode
    tokens.append(sample['processed_tokens'])#still unicode
    imgpaths.append(sample['file_path'])
    labels.append(sample['id']-1)

num_cap = 0
for items in captions:
    for item in items:
        num_cap += 1
print("Caption num:{}".format(num_cap))
print('Label num:{}'.format(len(labels)))

# '''
#     Resize image to 256*256 and save
# '''
# resizeimgdirpath = os.path.join(inputpath, 'imgs_256_python')
# if not os.path.exists(resizeimgdirpath):
#     print('resize image')
#     for i, imgpaths_i in enumerate(imgpaths):
#         img = Image.open(os.path.join(imgdirpath, imgpaths_i))
#         # img = img.resize((256, 256))
#         img = img.resize((384, 192))
#         resizeimgpath = os.path.join(resizeimgdirpath, '{}jpg'.format(imgpaths_i[:-3]))
#         if not os.path.exists(os.path.dirname(resizeimgpath)):
#             os.makedirs(os.path.dirname(resizeimgpath))
#         img.save(resizeimgpath, 'jpeg')
        
# '''
#     Calculate the mean of train image in the dataset
#     [R,G,B]
# '''
# imgmeanpath = os.path.join(outputdirpath, 'imgmean' + suffix)
# if not os.path.exists(imgmeanpath):
#     print('calculate imgmean')
#     imgmean = []
#     for i, imgpaths_i in enumerate(imgpaths):
#         img = Image.open(os.path.join(imgdirpath, imgpaths_i))
#         # img = img.resize((256, 256))
#         img = img.resize((384, 192))
#         if split[i] == 0:
#             imgmean.append(np.mean(np.mean(np.array(img, dtype=np.float), 0), 0))
#     imgmean = np.mean(np.vstack(imgmean), 0) / 255
#     np.save(imgmeanpath, imgmean)

# region
'''
    Generate vectors of words
    It can be replaced by the Bert
'''
'''
datasetdictpath = os.path.join(outputdirpath, 'dictionary.npy')
if os.path.exists(datasetdictpath):
    print('load dictionary')
    datasetdict = np.load(datasetdictpath, allow_pickle=True).item()
    dataset_words = datasetdict['dataset_words']
    dataset_vectors = datasetdict['dataset_vectors']
else:
    print('build dictionary')
    googlenews_words = sio.loadmat(googlenews_words_path)['w_names'][0]
    googlenews_words = {word[0].encode('utf-8'): i for i, word in enumerate(googlenews_words)}
    googlenews_vectors = h5py.File(googlenews_vectors_path, 'r')['w_features']
    dataset_words = {'<pad>': 0}
    dataset_vectors = np.zeros((googlenews_vectors.shape))#let <pad> vector be zeros
    dataset_index = 1
    if case == 'case':
        for i, captions_i in enumerate(captions):
            if split[i] == 0:
                for captions_i_j in captions_i:
                    for captions_i_j_k in filter(None, re.split(r'[-\s.,\(\)]+', captions_i_j.encode('utf-8'))):
                        if not dataset_words.has_key(captions_i_j_k):
                            currentindex = googlenews_words.get(captions_i_j_k)#python dict.get has the function of hash
                            if currentindex is not None:
                                dataset_words[captions_i_j_k] = dataset_index#compare to matlab version, ['Spongebob', 'Leo', 'Be', 'south', 'lowlights'] are included. haven't investigated why
                                dataset_vectors[dataset_index, :] = googlenews_vectors[currentindex, :]
                                dataset_index = dataset_index + 1
        dataset_vectors = dataset_vectors[:dataset_index, :]
    elif case == 'lowercase':
        for i, tokens_i in enumerate(tokens):
            if split[i] == 0:
                for tokens_i_j in tokens_i:
                    for tokens_i_j_k in tokens_i_j:
                        if not dataset_words.has_key(tokens_i_j_k):
                            currentindex = googlenews_words.get(tokens_i_j_k)#python dict.get has the function of hash
                            if currentindex is not None:
                                dataset_words[tokens_i_j_k] = dataset_index
                                dataset_vectors[dataset_index, :] = googlenews_vectors[currentindex, :]
                                dataset_index = dataset_index + 1
        dataset_vectors = dataset_vectors[:dataset_index, :]
    else:
        raise Exception('case wrong')
    np.save(datasetdictpath, {'dataset_words': dataset_words, 'dataset_vectors': dataset_vectors})
'''
# endregion
datasetdictpath = os.path.join(outputdirpath, 'dictionary' + suffix)
if os.path.exists(datasetdictpath):
    print('load dictionary')
    datasetdict = np.load(datasetdictpath, allow_pickle=True).item()
    dataset_words = datasetdict['dataset_words']
    dataset_vectors = datasetdict['dataset_vectors']
else:
    print('build dictionary')
    # 9408 unique words in dataset
    # {words(str):index}
    dataset_words = {}
    # np, size:[word_index, vectors_length]
    dataset_vectors = np.zeros((10000, dim))
    # # record word idnex, for save the vectors in dataset_vectors
    word_index = 1
    if case == 'case':
        # 根据captions 构建两个字典
        # dataset_words和dataset_vectors
        for i, captions_i in enumerate(captions):
            if i % 1000 == 0:
                print(i)
            for captions_i_j in captions_i:
                # transfer caption to words
                words_in_captions = sent2words(captions_i_j)
                for captions_i_j_k in words_in_captions:
                    if captions_i_j_k in dataset_words.keys():
                        continue
                    # tokens for single word
                    tokens_i_j_k, masks_i_j_k = str2tokens(captions_i_j_k, maxlength, tokenizer)
                    # use token to get vectors by bert
                    vectors_i_j_k = tokens2vec(tokens_i_j_k, masks_i_j_k, bert, device)

                    tokens = tokens_i_j_k[0][1]

                    dataset_words[captions_i_j_k] = word_index
                    dataset_vectors[word_index, :] = vectors_i_j_k
                    word_index = word_index + 1
        dataset_vectors = dataset_vectors[:word_index, :]
    np.save(datasetdictpath, {'dataset_words': dataset_words, 'dataset_vectors': dataset_vectors})

'''
    sentence -> index
'''
def sent2idx(sentence, maxlength):
    words_idx = torch.zeros(maxlength, dtype=torch.int64)

    words = sent2words(sentence)
    if len(words) > 80:
        print(len(words))
    for i, word in enumerate(words[:maxlength]):
        words_idx[i] = dataset_words[word]
    return words_idx

'''
    添加样本
'''
# imglabel_i用于标注两个句子描述同一个图像
def addsample_case(dataset, imgpaths_i, captions_i, labels_i, imglabel_i, maxlength):
    dataset['imgpath'].append('{}jpg'.format(imgpaths_i[:-3]))
    dataset['imgid'].append(labels_i)

    # some images have more than two sentences
    # here we make a sentences limitation
    for captions_i_j in captions_i[:sent_per_img]:

        # tokens, masks = str2tokens(captions_i_j, maxlength, tokenizer)
        # txtword这一个键值中，存储的应该是指向与word对应的vector的位置
        words_idx = sent2idx(captions_i_j, maxlength) 
        if words_idx[0] == 0:
            print('empty')
            # 不能跳过，为空也要写入，不然会影响后续读取

        dataset['txtword'].append(words_idx)
        dataset['txtid'].append(labels_i)
        dataset['imglabel'].append(imglabel_i)
        dataset['captions'].append(captions_i_j)
def addsample_case_pair(dataset, imgpaths_i, captions_i, labels_i, imglabel_i, maxlength):
    # 这里的封装方式和origin的不一样，这里采用了保证了image-text对应关系
    # 而不是和之前一样选取一个同id的文本进行训练
    for captions_i_j in captions_i:
        sample = []
        sample.append('{}jpg'.format(imgpaths_i[:-3]))
        
        words_index = sent2idx(captions_i_j, maxlength)

        sample.append(words_index)
        sample.append(labels_i)
        sample.append(imglabel_i)
        dataset.append(sample)
# region
# def addsample_lowercase(dataset, imgpaths_i, tokens_i, labels_i, maxlength):
#     dataset['imgpath'].append('{}jpg'.format(imgpaths_i[:-3]))
#     dataset['imgid'].append(labels_i)
#     for tokens_i_j in tokens_i:
#         txtword = np.zeros((maxlength), dtype=np.int16)
#         l = 0
#         for tokens_i_j_k in tokens_i_j:
#             currentindex = dataset_words.get(tokens_i_j_k)
#             if currentindex is not None:
#                 txtword[l] = currentindex
#                 l = l + 1
#                 if l >= maxlength:
#                     break
#         dataset['txtword'].append(txtword)
#         dataset['txtid'].append(labels_i)

# def addsample_lowercase_pair(dataset, imgpaths_i, tokens_i, labels_i, maxlength):
#     for tokens_i_j in tokens_i:
#         sample = []
#         sample.append('{}jpg'.format(imgpaths_i[:-3]))
#         txtword = np.zeros((maxlength), dtype=np.int16)
#         l = 0
#         for tokens_i_j_k in tokens_i_j:
#             currentindex = dataset_words.get(tokens_i_j_k)
#             if currentindex is not None:
#                 txtword[l] = currentindex
#                 l = l + 1
#                 if l >= maxlength:
#                     break
#         sample.append(txtword)
#         sample.append(labels_i)
#         dataset.append(sample)
# endregion

dataset_train = {'imgpath': [], 'imgid': [], 'txtword': [], 'txtid': [], 'imglabel': [], 'captions': []}
dataset_train_pair = []
dataset_val = {'imgpath': [], 'imgid': [], 'txtword': [], 'txtid': [], 'imglabel': [], 'captions': []}
dataset_test = {'imgpath': [], 'imgid': [], 'txtword': [], 'txtid': [], 'imglabel': [], 'captions': []}
if case == 'case':
    for i in range(len(split)):
        if i%1000 == 0:
            print(i)
        if split[i] == 0:
            addsample_case(dataset_train, imgpaths[i], captions[i], labels[i], i, maxlength)
            addsample_case_pair(dataset_train_pair, imgpaths[i], captions[i], labels[i], i, maxlength)
        elif split[i] == 1:
            # print('add test')
            addsample_case(dataset_val, imgpaths[i], captions[i], labels[i], i, maxlength)
        else:
            # print('add val')
            addsample_case(dataset_test, imgpaths[i], captions[i], labels[i], i, maxlength)
# elif case == 'lowercase':
#     for i in range(len(split)):
#         if split[i] == 0:
#             addsample_lowercase(dataset_train, imgpaths[i], tokens[i], labels[i], maxlength)
#             addsample_lowercase_pair(dataset_train_pair, imgpaths[i], tokens[i], labels[i], maxlength)
#         elif split[i] == 1:
#             addsample_lowercase(dataset_val, imgpaths[i], tokens[i], labels[i], maxlength)
#         else:
#             addsample_lowercase(dataset_test, imgpaths[i], tokens[i], labels[i], maxlength)
else:
    raise Exception('case wrong')
        

def listtoarray(dataset, phase):
    dataset['imgid'] = np.array(dataset['imgid'], dtype=np.int64)
    dataset['txtword'] = np.vstack(dataset['txtword'])
    dataset['txtid'] = np.array(dataset['txtid'], dtype=np.int64)
    dataset['imglabel'] = np.array(dataset['imglabel'], dtype=np.int64)
    np.save(os.path.join(outputdirpath, phase+suffix), dataset)

# train.npy => dict_keys(['txtword', 'txtid', 'imgid', 'imgpath'])
listtoarray(dataset_train, 'train')
# train_pair.npy list([image_path, txt_id, id])
# np.save(os.path.join(outputdirpath, 'train'+'_pair'+ suffix), dataset_train_pair)
# val.npy => dict_keys(['txtword', 'txtid', 'imgid', 'imgpath'])
listtoarray(dataset_val, 'val')
# test.npy => dict_keys(['txtword', 'txtid', 'imgid', 'imgpath'])
listtoarray(dataset_test, 'test')

