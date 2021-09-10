# extreactor features
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from torch.utils.data import Dataset
from PIL import Image
import pdb
import sys
sys.path.append('.')
from dataset import *
from module.model import Network
from config import cfg
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpuid',default='7', type=str,help='gpuid: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='best', type=str, help='0,1,2,3...or last')
parser.add_argument('--name', default='ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--imgdir_path', default='/home/wuziqiang/data/CUHK-PEDES/CUHK_PEDES_prepare/imgs_256_python', type=str)
parser.add_argument('--config_file', default='./config/configs_CUHKPEDES.yaml', type=str, help='config parameters')

opt = parser.parse_args()

cfg.merge_from_file(opt.config_file)
cfg.freeze()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid

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
        return img, imgid, self.dataset['imgpath'][idx]
        
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

# Load data
transform_gallery_list = [
        transforms.Resize(size=(384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
transform_query_list = [
        ]
data_transforms = {
    'gallery': transforms.Compose(transform_gallery_list),
    'query': transforms.Compose(transform_query_list),
}
use_gpu = torch.cuda.is_available()

dataset_test_path = 'dataset/cuhkpedes_case_maxlen=80_768_stopwords_transtest/test.npy'
dictionary_path = 'dataset/cuhkpedes_case_maxlen=80_768_stopwords_transtest/dictionary.npy'
imgdir_path = opt.imgdir_path

dataset_query = DatasetQuery(dataset_test_path, dictionary_path, transform=data_transforms['query'])
dataset_gallery = DatasetGallery(dataset_test_path, imgdir_path, transform=data_transforms['gallery'])

dataloaders_query = torch.utils.data.DataLoader(dataset_query, batch_size=int(opt.batchsize/8), shuffle=False, num_workers=8)
dataloaders_gallery = torch.utils.data.DataLoader(dataset_gallery, batch_size=opt.batchsize, shuffle=False, num_workers=8) 

###################
#  Load model
###################
def load_network(network):      
    baseline_path = '/data/wuziqiang/model/CUHKPEDES/B_baseline1_CUHKPEDES_resnet50_(384,128)_T=1.0_LMD1=1.0_LMD2=3.0_LMD3=16.0_LMD4=1.0_RKT/best.pth.tar'
    mh_loss_path = '/data/wuziqiang/model/CUHKPEDES/fortestingNone_CUHKPEDES_resnet50_(384,128)_T=10.0_LMD1=1.0_LMD2=3.0_LMD3=16.0_LMD4=1.0_RKT/best.pth.tar'
    network.load_state_dict(torch.load(mh_loss_path)['model'])
    return network

def fliplr(img):
    '''flip horizontal'''
    # 倒着排序
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
    # 对dim=3进行倒序选取
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature_gallery(model, loader):
    features = torch.FloatTensor()
    gallery_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        img, label, imgpath = data
        gallery_label = torch.cat((gallery_label, label))
        ff = []
        for i in range(2):
            # 添加图像flip操作
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            outputs = model.visual_model(input_img)    
            outputs = model.embed_model.avgpool(outputs)  
            batch_size = outputs.size(0)
            outputs = outputs.view(batch_size, -1)
            outputs = model.embed_model.visual_embed_layer(outputs)
            outputs = model.embed_model.bottelneck_global_visual(outputs)

            ff.append(outputs.data.cpu())
        ff = ff[0] + ff[1]
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features, gallery_label

def extract_feature_query(model, loader):
    features = torch.FloatTensor()
    query_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        txt, label, txtlen = data
        query_label = torch.cat((query_label, label))
        input_txt = Variable(txt.cuda())
        input_txtlen = Variable(txtlen.cuda())
        batch_size = input_txt.size(0)

        outputs = model.textual_model(input_txt, input_txtlen)
        outputs = outputs.view(batch_size, -1)
        outputs = model.embed_model.textual_embed_layer(outputs)
        outputs = model.embed_model.bottelneck_global_textual(outputs)
        ff = outputs.data.cpu()
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features, query_label

print('-'*10, 'test', '-'*10)

network_structure = Network(cfg, 11003)
model = load_network(network_structure)

model = model.eval()
if use_gpu:
    model = model.cuda()
    print('using gpu')

query_feature, query_label = extract_feature_query(model, dataloaders_query)
gallery_feature, gallery_label = extract_feature_gallery(model, dataloaders_gallery)
print(query_label)
print(gallery_label)
result = {'query_f': query_feature.numpy(),
            'query_label': query_label.numpy(),
            'gallery_f': gallery_feature.numpy(),
            'gallery_label': gallery_label.numpy()}

savepath = './evaluate/result/result_mhloss_refiner.mat'
# savepath = './evaluate/result/result_baseline.mat'
if not os.path.exists('/'.join(savepath.split('/')[:-1])):
    os.makedirs('/'.join(savepath.split('/')[:-1]))
scipy.io.savemat(savepath, result)