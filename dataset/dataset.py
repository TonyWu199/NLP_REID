import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import math

######################################################################
# Load Data
class DatasetTrain(Dataset):
    def __init__(self, dataset_path, dictionary_path, imgdir_path, transform=None, datasetname='cuhkpedes'):
        # self.dataset = np.load(dataset_path, allow_pickle=True).item()
        # self.dataset_vectors = np.load(dictionary_path, allow_pickle=True).item()['dataset_vectors']
        self.dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1').item()
        self.dataset_vectors = np.load(dictionary_path, allow_pickle=True, encoding='latin1').item()['dataset_vectors']
        self.imgdir_path = imgdir_path
        self.transform = transform
        self.maxlength = self.dataset['txtword'].shape[1]
        self.embeddingdim = self.dataset_vectors.shape[1]
        # self.embeddingdim = 3072  # for bert embeddings
        self.datasetname = datasetname
        self.sent_per_img = int(len(self.dataset['txtword']) / len(self.dataset['imgpath']))

    # index range
    def __len__(self):
        return len(self.dataset['imgpath'])
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgdir_path, self.dataset['imgpath'][idx]))
        if self.transform:
            img = self.transform(img)
        imgid = self.dataset['imgid'][idx]
        imglabel = self.dataset['imglabel'][idx]

        # Strict：返回和img对应的两段描述，
        # unstrict：返回和img id对应的一段描述
        # if self.method == 'cls':
        
        # 从所有与img相同id的txt中随机抽取一个
        # *select idy
        # idy = np.random.choice(np.argwhere(self.dataset['txtid'] == imgid)[:, 0])
        # *选择与image对应的txt
        idy = np.random.choice(range(idx*self.sent_per_img, (idx+1)*self.sent_per_img, 1)) 

        imgid = torch.from_numpy(np.array(imgid))
        txt, txtid, txtlen = self.get_vector(idy)

        return img, imgid, txt, txtid, txtlen, imglabel

    def get_vector(self, idy):
        # size(id(int)) = max_length
        txtword = self.dataset['txtword'][idy, :]
        txt = np.zeros([self.maxlength, self.embeddingdim], dtype = np.float32)
        # word_embedding [maxlength, dim(embeddings of a word)]
        for i, txtword_i in enumerate(txtword):
            if txtword_i <= 0:
                break
            txt[i, :] = self.dataset_vectors[txtword_i, :]
        txtlen = i
        txt = torch.from_numpy(txt)
        txtid = torch.from_numpy(np.array(self.dataset['txtid'][idy])) 
        return txt, txtid, txtlen 



class DatasetTrain_Triplet(Dataset):
    def __init__(self, dataset_path, dictionary_path, imgdir_path, transform=None):
        self.dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1').item()
        self.dataset_vectors = np.load(dictionary_path, allow_pickle=True, encoding='latin1').item()['dataset_vectors']
        self.imgdir_path = imgdir_path
        self.transform = transform
        self.maxlength = self.dataset['txtword'].shape[1]
        self.embeddingdim = self.dataset_vectors.shape[1]


        img_path_list = self.dataset['imgpath']
        img_id_list = self.dataset['imgid']
        txt_word_list = self.dataset['txtword']
        txt_id_list = self.dataset['txtid']
        # 构建字典，降低样本选择时的时间复杂度
        # {id:[[img_path], [txt word]]}
        self.id_dict = {}
        for i in range(len(img_path_list)):
            img_id = img_id_list[i]
            # img_id在字典键值中
            if img_id in self.id_dict:
                self.id_dict[img_id][0].append(img_path_list[i])
                self.id_dict[img_id][1].append(txt_word_list[i])
            else:
                self.id_dict[img_id] = [[img_path_list[i]], [txt_word_list[i]]] 
        
    def __len__(self):
        return len(self.dataset['imgpath'])

    # 原先直接采用遍历self.dataset['imgpath']的方式进行triplet样本选择，导致一个epoch耗时30min
    # 改进后采用 dict{id:[[imgpath], [txtword]]}的数据结构进行存储，因为主要是对id的筛选，因此这样使得速度提升很多
    # Tips:两张卡并行比单独用一张卡慢
    def __getitem__(self, idx):
        '''
            Anchor
        '''
        anchor_idx = idx
        anchor_img_path, anchor_img, anchor_img_id, anchor_txt, anchor_txt_id, anchor_txt_len = self.get_data(anchor_idx)

        '''
            Positive
        '''
        # choice_list = [x for x in range(len(self.dataset['imgpath'])) \
        #     if self.dataset['imgid'][x] == anchor_img_id and os.path.join(self.imgdir_path, self.dataset['imgpath'][x]) != anchor_img_path]
        # pos_idx = idx if len(choice_list)==0 else np.random.choice(choice_list)
        # pos_img_path, pos_img, pos_img_id, pos_txt, pos_txt_id, pos_txt_len = self.get_data(pos_idx)
        choice_pos_img_list = [os.path.join(self.imgdir_path, path) for path in self.id_dict[anchor_img_id.item()][0] \
                            if os.path.join(self.imgdir_path, path) != anchor_img_path]
        pos_img = anchor_img_path if len(choice_pos_img_list)==0 else np.random.choice(choice_pos_img_list)
        pos_img = Image.open(pos_img)
        pos_img_id = anchor_img_id
        if self.transform:
            pos_img = self.transform(pos_img)

        choice_pos_txt_list = self.id_dict[anchor_img_id.item()][1]
        # choice_txt_list为二维数组，random.choice仅适用于一维数组
        pos_txt_word = choice_pos_txt_list[np.random.randint(len(choice_pos_txt_list))]
        pos_txt_id = anchor_txt_id
        pos_txt = np.zeros([self.maxlength, self.embeddingdim], dtype = np.float32)
        for i, txt_word_i in enumerate(pos_txt_word):
            if txt_word_i <= 0:
                break
            pos_txt[i, :] = self.dataset_vectors[txt_word_i, :]
        pos_txt_len = i
        pos_txt = torch.from_numpy(pos_txt)

        '''
            Negative
        '''
        # neg_idx = np.random.choice([x for x in range(len(self.dataset['imgpath']))\
        #     if self.dataset['imgid'][x] != anchor_img_id])
        # neg_img_path, neg_img, neg_img_id, neg_txt, neg_txt_id, neg_txt_len = self.get_data(neg_idx)
        choice_neg_id_list = [i for i in self.id_dict.keys() if i != anchor_img_id.item()]
        neg_id = np.random.choice(choice_neg_id_list)
        choice_neg_img_list = [os.path.join(self.imgdir_path, path) for path in self.id_dict[neg_id][0]]
        neg_img = np.random.choice(choice_neg_img_list)
        neg_img = Image.open(neg_img)
        neg_img_id = torch.from_numpy(np.array(neg_id))
        if self.transform:
            neg_img = self.transform(neg_img)

        choice_neg_txt_list = self.id_dict[neg_id][1]
        neg_txt_word = choice_neg_txt_list[np.random.randint(len(choice_neg_txt_list))]
        neg_txt_id = neg_img_id
        neg_txt = np.zeros([self.maxlength, self.embeddingdim], dtype = np.float32)
        for i, txt_word_i in enumerate(neg_txt_word):
            if txt_word_i <= 0:
                break
            neg_txt[i, :] = self.dataset_vectors[txt_word_i, :]
        neg_txt_len = i
        neg_txt = torch.from_numpy(neg_txt)

        return anchor_img, anchor_img_id, anchor_txt, anchor_txt_id, anchor_txt_len, \
                pos_img, pos_img_id, pos_txt, pos_txt_id, pos_txt_len, \
                    neg_img, neg_img_id, neg_txt, neg_txt_id, neg_txt_len

    def get_data(self, idx):
        img_path = os.path.join(self.imgdir_path, self.dataset['imgpath'][idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        img_id = self.dataset['imgid'][idx]

        # 原有的训练方式是从所有与img相同id的txt中随机抽取一个
        idy = np.random.choice(np.argwhere(self.dataset['txtid'] == img_id)[:, 0])
        img_id = torch.from_numpy(np.array(img_id))

        # words embedding
        txt_word = self.dataset['txtword'][idy, :]
        txt = np.zeros([self.maxlength, self.embeddingdim], dtype = np.float32)
        for i, txtword_i in enumerate(txt_word):
            if txtword_i <= 0:
                break
            txt[i, :] = self.dataset_vectors[txtword_i, :]
        txt_len = i
        txt = torch.from_numpy(txt)
        txt_id = torch.from_numpy(np.array(self.dataset['txtid'][idy]))   

        return img_path, img, img_id, txt, txt_id, txt_len    


class DatasetGallery(Dataset):
    def __init__(self, dataset_path, imgdir_path, transform=None):
        # self.dataset = np.load(dataset_path, allow_pickle=True).item()
        self.dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1').item()
        self.imgdir_path = imgdir_path
        self.transform = transform
    def __len__(self):
        return len(self.dataset['imgpath'])
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgdir_path, self.dataset['imgpath'][idx]))
        if self.transform:
            img = self.transform(img)
        imgid = self.dataset['imgid'][idx]
        imgid = torch.from_numpy(np.array(imgid))
        return img, imgid
        
class DatasetQuery(Dataset):
    def __init__(self, dataset_path, dictionary_path, transform=None):
        # self.dataset = np.load(dataset_path, allow_pickle=True).item()
        # self.dataset_vectors = np.load(dictionary_path, allow_pickle=True).item()['dataset_vectors']
        self.dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1').item()
        self.dataset_vectors = np.load(dictionary_path, allow_pickle=True, encoding='latin1').item()['dataset_vectors']
        self.maxlength = self.dataset['txtword'].shape[1]
        self.embeddingdim = self.dataset_vectors.shape[1]
    def __len__(self):
        return self.dataset['txtword'].shape[0]
    def __getitem__(self, idx):
        txtword = self.dataset['txtword'][idx, :]
        txt = np.zeros([self.maxlength, self.embeddingdim], dtype = np.float32)
        for i, txtword_i in enumerate(txtword):
            if txtword_i <= 0:
                break
            txt[i, :] = self.dataset_vectors[txtword_i, :]
        txtlen = i
        txt = torch.from_numpy(txt)
        txtid = torch.from_numpy(np.array(self.dataset['txtid'][idx]))
        return txt, txtid, txtlen