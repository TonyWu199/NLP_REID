# -*- coding: utf-8 -*-

from __future__ import print_function, division

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
from model_stage2 import ft_net_stage2_double

from torch.utils.data import Dataset
from PIL import Image
import pdb

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='3', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zzd/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50_2', type=str, help='save model path')
parser.add_argument('--batchsize', default=50, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )

opt = parser.parse_args()

T = 2
w_soft_1 = 3.0
w_soft_2 = 0.0
w_soft_3 = 16.0
student = '1'
optimchoose = 'adam'
lrmul = 0.1

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name + '_T' + str(T) + 'wsoft1' + str(w_soft_1) + 'wsoft2' + str(w_soft_2) + 'wsoft3' + str(w_soft_3) + 'student' + str(student) + 'optimchoose' + str(optimchoose) + 'lrmul' + str(lrmul)
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
transform_gallery_list = [
        transforms.Resize(size=(224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.3907, 0.3668, 0.3502], [1, 1, 1])
        ]
transform_query_list = [
        ]
data_transforms = {
    'gallery': transforms.Compose( transform_gallery_list ),
    'query': transforms.Compose(transform_query_list),
}

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


class CuhkpedesDataset_gallery(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, cuhkpedes_filename, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cuhkpedes = scipy.io.loadmat(os.path.join('/mnt/local0/chenyucheng/code/code/textimgreid/textimgreid/dualpathITE/dataset_newserver', cuhkpedes_filename + '.mat'))[cuhkpedes_filename]
        self.transform = transform

    def __len__(self):
        return len(self.cuhkpedes['imgpath'][0, 0].T)

    def __getitem__(self, idx):
        img = Image.open(self.cuhkpedes['imgpath'][0, 0][0, idx][0].encode('unicode-escape'))
        if self.transform:
            img = self.transform(img)
        imgid = self.cuhkpedes['imgid'][0, 0][0, idx]
        imgid = torch.from_numpy(np.array(imgid - 1, dtype = np.int64))
        return img, imgid
class CuhkpedesDataset_query(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, cuhkpedes_filename, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cuhkpedes = scipy.io.loadmat(os.path.join('/mnt/local0/chenyucheng/code/code/textimgreid/textimgreid/dualpathITE/dataset_newserver', cuhkpedes_filename + '.mat'))[cuhkpedes_filename]
        self.cuhkpedes_dictionary = scipy.io.loadmat('/mnt/local0/chenyucheng/code/code_download/textimgreid/dualpathITE/Image-Text-Embedding/dataset/CUHK-PEDES-prepare/CUHK-PEDES_dictionary.mat')['subset'][0,0][2]

    def __len__(self):
        return len(self.cuhkpedes['txtword'][0, 0].T)

    def __getitem__(self, idx):
        txt = np.zeros([56, 300])
        for j in range(56):
            v = self.cuhkpedes['txtword'][0, 0][j, idx]
            if v <= 0:
                break
            txt[j, :] = self.cuhkpedes_dictionary[:, v - 1]
        txt = torch.from_numpy(np.array(txt, dtype = np.float32))
        txtid = torch.from_numpy(np.array(self.cuhkpedes['txtid'][0, 0][0, idx] - 1, dtype = np.int64))
        txtlen = j
        return txt, txtid, txtlen

image_datasets = {}
image_datasets['gallery'] = CuhkpedesDataset_gallery(cuhkpedes_filename='cuhkpedes_test', transform=data_transforms['gallery'])
image_datasets['query'] = CuhkpedesDataset_query(cuhkpedes_filename='cuhkpedes_test', transform=data_transforms['query'])
batchsize = {}
batchsize['gallery'] = opt.batchsize
batchsize['query'] = int(opt.batchsize / 8)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize[x],
                                         shuffle=False, num_workers=8) for x in ['gallery','query']}
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature_gallery(model,dataloaders):
    features = torch.FloatTensor()
    gallery_label = np.array([], dtype = np.int64)
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        gallery_label = np.append(gallery_label, label.numpy())
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,1024).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())

            outputs = model.embeddingnet.model_pretrain_stage1.model.conv1(input_img)
            outputs = model.embeddingnet.model_pretrain_stage1.model.bn1(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.relu(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.maxpool(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.layer1(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.layer2(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.layer3(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.layer4(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.conv1x1_img(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.model.avgpool(outputs)
            outputs = torch.squeeze(outputs)
            outputs = model.embeddingnet.model_pretrain_stage1.add_block_1.add_block(outputs)

            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features, gallery_label

def extract_feature_query(model,dataloaders):
    features = torch.FloatTensor()
    query_label = np.array([], dtype = np.int64)
    count = 0
    for data in dataloaders:
        txt, label, txtlen = data
        n, _, _ = txt.size()
        count += n
        print(count)
        query_label = np.append(query_label, label.numpy())
        ff = torch.FloatTensor(n,1024).zero_()
        input_txt = Variable(txt.cuda())
        input_txtlen = Variable(txtlen.cuda())

        outputs = model.embeddingnet.model_pretrain_stage1.txt_block_1.bilstm(input_txt, input_txtlen)
        outputs, _ = torch.max(outputs, dim=1)
        outputs = model.embeddingnet.model_pretrain_stage1.txt_block_1.fc(outputs)

        ff = outputs.data.cpu()
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features, query_label

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(11003)
else:
    model_structure = ft_net_stage2_double(11003)

if opt.PCB:
    model_structure = PCB(11003)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if not opt.PCB:
    model = model
else:
    model = PCB_test(model)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature, gallery_label = extract_feature_gallery(model,dataloaders['gallery'])
query_feature, query_label = extract_feature_query(model,dataloaders['query'])
if opt.multi:
    mquery_feature = extract_feature(model,dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
scipy.io.savemat('result/pytorch_result_stage2_epochlast.mat' + '_T' + str(T) + 'wsoft1' + str(w_soft_1) + 'wsoft2' + str(w_soft_2) + 'wsoft3' + str(w_soft_3) + 'student' + str(student) + 'optimchoose' + str(optimchoose) + 'lrmul' + str(lrmul),result)
if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)
