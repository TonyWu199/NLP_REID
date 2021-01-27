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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sent_per_img = 5
print('==>initiate bert tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert = bert().to(device)

#setting
parser = argparse.ArgumentParser()
parser.add_argument('--case', default='case', type=str)#not implement
parser.add_argument('--maxlength', default=32, type=int)
parser.add_argument('--del_stopwords', default=0, type=int)
opt = parser.parse_args()
print(opt)
case = opt.case
maxlength = opt.maxlength

#input
datasetname = 'flickr30k'
inputpath = '/home/wuziqiang/data/Flickr30K'
jsonpath = os.path.join(inputpath, 'dataset_flickr30k_29000.json')
imgdirpath = os.path.join(inputpath, 'flickr30k-images')
print(imgdirpath)
outputdirpath = '{}_{}_maxlen={}_stopwords'.format(datasetname, case, maxlength)
if not os.path.exists(outputdirpath):
    os.mkdir(outputdirpath)
with open(jsonpath, 'r') as f:
    jsondata = json.load(f)['images']
captions = []
tokens = []
imgpaths = []
labels = []
split = []
for sample in jsondata:
    captions.append([x['raw'] for x in sample['sentences']])#still unicode
    tokens.append([x['tokens'] for x in sample['sentences']])#still unicode
    imgpaths.append(sample['filename'])
    labels.append(sample['imgid'])
    if sample['split'] == 'train':
        split.append(0)
    elif sample['split'] == 'val':
        split.append(1)
    elif sample['split'] == 'test':
        split.append(2)
    else:
        raise Exception('split wrong')
print(len(jsondata))
print(">"*20, "Dataset Info", "<"*20)
print("Train Set : ", split.count(0))
print("Val Set : ", split.count(1))
print("Test Set : ", split.count(2))

resizeimgdirpath = os.path.join(inputpath, 'flickr30k_images_256')
if not os.path.exists(resizeimgdirpath):
    print('resize image')
    for i, imgpaths_i in enumerate(imgpaths):
        img = Image.open(os.path.join(imgdirpath, imgpaths_i))
        img = img.resize((256, 256))
        resizeimgpath = os.path.join(resizeimgdirpath, '{}jpg'.format(imgpaths_i[:-3]))
        if not os.path.exists(os.path.dirname(resizeimgpath)):
            os.makedirs(os.path.dirname(resizeimgpath))
        img.save(resizeimgpath, 'jpeg')

imgmeanpath = os.path.join(outputdirpath, 'imgmean.npy')
if not os.path.exists(imgmeanpath):
    print('calculate imgmean')
    imgmean = []
    for i, imgpaths_i in enumerate(imgpaths):
        img = Image.open(os.path.join(imgdirpath, imgpaths_i))
        img = img.resize((256, 256))
        if split[i] == 0:
            imgmean.append(np.mean(np.mean(np.array(img, dtype=np.float), 0), 0))
    imgmean = np.mean(np.vstack(imgmean), 0) / 255
    np.save(imgmeanpath, imgmean)

datasetdictpath = os.path.join(outputdirpath, 'dictionary.npy')
if os.path.exists(datasetdictpath):
    print('load dictionary')
    datasetdict = np.load(datasetdictpath, allow_pickle=True).item()
    dataset_words = datasetdict['dataset_words']
    dataset_vectors = datasetdict['dataset_vectors']
else:
    # print('build dictionary')
    # googlenews_words = sio.loadmat(googlenews_words_path)['w_names'][0]
    # googlenews_words = {word[0].encode('utf-8'): i for i, word in enumerate(googlenews_words)}
    # googlenews_vectors = h5py.File(googlenews_vectors_path, 'r')['w_features']
    # dataset_words = {'<pad>': 0}
    # dataset_vectors = np.zeros((googlenews_vectors.shape))#let <pad> vector be zeros
    # dataset_index = 1
    # if case == 'case':
    #     for i, captions_i in enumerate(captions):
    #         if split[i] == 0:
    #             for captions_i_j in captions_i:
    #                 for captions_i_j_k in filter(None, re.split(r'[-\s.,\(\)]+', captions_i_j.encode('utf-8'))):
    #                     if not dataset_words.has_key(captions_i_j_k):
    #                         currentindex = googlenews_words.get(captions_i_j_k)#python dict.get has the function of hash
    #                         if currentindex is not None:
    #                             dataset_words[captions_i_j_k] = dataset_index#compare to matlab version, ['Spongebob', 'Leo', 'Be', 'south', 'lowlights'] are included. haven't investigated why
    #                             dataset_vectors[dataset_index, :] = googlenews_vectors[currentindex, :]
    #                             dataset_index = dataset_index + 1
    #     dataset_vectors = dataset_vectors[:dataset_index, :]
    # elif case == 'lowercase':
    #     for i, tokens_i in enumerate(tokens):
    #         if split[i] == 0:
    #             for tokens_i_j in tokens_i:
    #                 for tokens_i_j_k in tokens_i_j:
    #                     if not dataset_words.has_key(tokens_i_j_k):
    #                         currentindex = googlenews_words.get(tokens_i_j_k)#python dict.get has the function of hash
    #                         if currentindex is not None:
    #                             dataset_words[tokens_i_j_k] = dataset_index
    #                             dataset_vectors[dataset_index, :] = googlenews_vectors[currentindex, :]
    #                             dataset_index = dataset_index + 1
    #     dataset_vectors = dataset_vectors[:dataset_index, :]
    # else:
    #     raise Exception('case wrong')
    # np.save(datasetdictpath, {'dataset_words': dataset_words, 'dataset_vectors': dataset_vectors})
    print('build dictionary')
    dataset_words = {}
    dataset_vectors = np.zeros((20000, 768))
    word_index = 1
    if case == 'case':
        for i, captions_i in enumerate(captions):
            if i < 1000:
                print(i)
            elif i % 1000==0:
                print(i)
            # at least 5 captions_i_j
            for captions_i_j in captions_i:
                # print(captions_i_j)
                words_in_captions = sent2words(captions_i_j, del_stopwords=opt.del_stopwords)
                for captions_i_j_k in words_in_captions:
                    if captions_i_j_k in dataset_words.keys():
                        continue
                    tokens_i_j_k, masks_i_j_k = str2tokens(captions_i_j_k, maxlength, tokenizer)
                    vectors_i_j_k = tokens2vec(tokens_i_j_k, masks_i_j_k, bert, device)

                    tokens = tokens_i_j_k[0][1]
                    dataset_words[captions_i_j_k] = word_index
                    dataset_vectors[word_index, :] = vectors_i_j_k
                    word_index = word_index + 1
        dataset_vectors = dataset_vectors[:word_index, :]
    np.save(datasetdictpath, {'dataset_words': dataset_words, 'dataset_vectors':dataset_vectors})

'''
    sentence -> index
'''
def sent2idx(sentence, maxlength):
    words_idx = torch.zeros(maxlength, dtype=torch.int64)

    words = sent2words(sentence, del_stopwords=opt.del_stopwords)
    if len(words) > 80:
        print(len(words))

    maxlength = len(words) if len(words) < maxlength else maxlength
    for i, word in enumerate(words[:maxlength]):
        words_idx[i] = dataset_words[word]
    if words_idx[0] == 0:
        print(sentence, words)
    return words_idx

def addsample_case(dataset, imgpaths_i, captions_i, labels_i, imglabel_i, maxlength):
    dataset['imgpath'].append('{}jpg'.format(imgpaths_i[:-3]))
    dataset['imgid'].append(labels_i)
    
    for captions_i_j in captions_i[:]:

        words_idx = sent2idx(captions_i_j, maxlength)

        dataset['txtword'].append(words_idx)
        dataset['txtid'].append(labels_i)
        dataset['imglabel'].append(imglabel_i)

def addsample_case_pair(dataset, imgpaths_i, captions_i, labels_i, imglabel_i, maxlength):
    for captions_i_j in captions_i:
        sample = []
        sample.append('{}jpg'.format(imgpaths_i[:-3]))

        words_index = sent2idx(captions_i_j, maxlength)
        
        sample.append(words_index)
        sample.append(labels_i)
        sample.append(imglabel_i)
        dataset.append(sample)

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
        
dataset_train = {'imgpath': [], 'imgid': [], 'txtword': [], 'txtid': [], 'imglabel': []}
dataset_train_pair = []
dataset_val = {'imgpath': [], 'imgid': [], 'txtword': [], 'txtid': [], 'imglabel': []}
dataset_test = {'imgpath': [], 'imgid': [], 'txtword': [], 'txtid': [], 'imglabel': []}
num = 0
if case == 'case':
    for i in range(len(split)):
        if i%1000 == 0:
            print(i)
        if split[i] == 0:
            addsample_case(dataset_train, imgpaths[i], captions[i], labels[i], i, maxlength)
            addsample_case_pair(dataset_train_pair, imgpaths[i], captions[i], labels[i], i, maxlength)
        elif split[i] == 1:
            addsample_case(dataset_val, imgpaths[i], captions[i], labels[i], i, maxlength)
        else:
            addsample_case(dataset_test, imgpaths[i], captions[i], labels[i], i, maxlength)
        num = i
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
    np.save(os.path.join(outputdirpath, phase+'.npy'), dataset)

train_path = os.path.join(outputdirpath, 'train.npy')
if not os.path.exists(train_path):
    print("Write train val test...")
    listtoarray(dataset_train, 'train')
    np.save(os.path.join(outputdirpath, 'train'+'_pair.npy'), dataset_train_pair)
    listtoarray(dataset_val, 'val')
    listtoarray(dataset_test, 'test')
