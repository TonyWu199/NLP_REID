from __future__ import print_function, division

import torch
from torch.autograd import Variable
import pdb
from module.utils import *
import logging

logger = logging.getLogger()  #root logger
logger.setLevel(logging.INFO)


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature_gallery(model, loader, cfg):
    features = torch.FloatTensor()
    gallery_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        img, label = data
        gallery_label = torch.cat((gallery_label, label))
        ff = []
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            # [batch_size, 1024]
            # 是否提取分块特征
            if cfg.MODEL.VISUAL_MODEL.NUM_STRIPES != 0:
                outputs, local = model.visual_model(input_img)
            else:
                outputs = model.visual_model(input_img)    
            # (feature, logits)
            # if 'PCB' in cfg.MODEL.VISUAL_MODEL_NAME:
            #     outputs = outputs[0]
            # if 'resnet' in cfg.MODEL.VISUAL_MODEL_NAME:
            #     outputs = model.embed_model.avgpool(outputs)
            outputs = model.embed_model.avgpool(outputs)  
            batch_size = outputs.size(0)
            outputs = outputs.view(batch_size, -1)
            outputs = model.embed_model.visual_embed_layer(outputs)
            if cfg.MODEL.BN_LAYER:
                outputs = model.embed_model.bottelneck_global_visual(outputs)

            ff.append(outputs.data.cpu())
        ff = ff[0] + ff[1]

        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features, gallery_label


def extract_feature_query(model, loader, cfg):
    features = torch.FloatTensor()
    query_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        txt, label, txtlen = data
        query_label = torch.cat((query_label, label))
        input_txt = Variable(txt.cuda())
        input_txtlen = Variable(txtlen.cuda())

        if cfg.MODEL.TEXTUAL_MODEL.WORDS:
            outputs, local = model.textual_model(input_txt, input_txtlen)
        else:
            outputs = model.textual_model(input_txt, input_txtlen)
        batch_size = input_txt.size(0)
        outputs = outputs.view(batch_size, -1)
        outputs = model.embed_model.textual_embed_layer(outputs)
        if cfg.MODEL.BN_LAYER:
            outputs = model.embed_model.bottelneck_global_textual(outputs)

        ff = outputs.data.cpu()

        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features, query_label


def test(model, loader_query, loader_gallery, cfg, avenorm=True):
    query_feature, query_label = extract_feature_query(model, loader_query, cfg)
    gallery_feature, gallery_label = extract_feature_gallery(model, loader_gallery, cfg)
    score = torch.mm(query_feature, gallery_feature.t())
    _, index_q2g = torch.sort(score, dim=1, descending=True)
    _, index_g2q = torch.sort(score, dim=0, descending=True)
    match_q2g = (query_label.reshape(
        -1, 1).expand_as(score) == gallery_label[index_q2g])
    match_g2q = (gallery_label.reshape(
        1, -1).expand_as(score) == query_label[index_g2q])
    cmc_q2g = torch.zeros((score.shape[1]))
    for i in range(score.shape[0]):
        cmc_i = torch.zeros((score.shape[1]))
        cmc_i[match_q2g[i, :].nonzero(as_tuple=False)[0][0]:] = 1
        cmc_q2g = cmc_q2g + cmc_i
    cmc_q2g = cmc_q2g / score.shape[0]
    cmc_g2q = torch.zeros((score.shape[0]))
    for i in range(score.shape[1]):
        cmc_i = torch.zeros((score.shape[0]))
        cmc_i[match_g2q[:, i].nonzero(as_tuple=False)[0][0]:] = 1
        cmc_g2q = cmc_g2q + cmc_i
    cmc_g2q = cmc_g2q / score.shape[1]

    query_feature_ave = torch.tensor([])
    query_label_ave = torch.tensor([], dtype=torch.int64)
    query_feature_temp = query_feature[0, :]
    sameidnum = 1.0
    for i in range(len(query_label) - 1):
        if query_label[i + 1] == query_label[i]:
            query_feature_temp = query_feature_temp + query_feature[i + 1, :]
            sameidnum = sameidnum + 1
        elif query_label[i + 1] > query_label[i]:
            query_feature_temp = query_feature_temp / sameidnum
            query_feature_ave = torch.cat(
                (query_feature_ave, query_feature_temp.reshape(1, -1)), 0)
            query_label_ave = torch.cat(
                (query_label_ave, query_label[i].unsqueeze(0)))
            query_feature_temp = query_feature[i + 1, :]
            sameidnum = 1.0
        else:
            raise Exception('label is not ascending')
    query_feature_temp = query_feature_temp / sameidnum
    query_feature_ave = torch.cat(
        (query_feature_ave, query_feature_temp.reshape(1, -1)), 0)
    query_label_ave = torch.cat(
        (query_label_ave, query_label[i + 1].unsqueeze(0)))
    if avenorm:
        fnorm = torch.norm(query_feature_ave, p=2, dim=1, keepdim=True)
        query_feature_ave = query_feature_ave.div(
            fnorm.expand_as(query_feature_ave))
    score_ave = torch.mm(query_feature_ave, gallery_feature.t())
    _, index_q2g_ave = torch.sort(score_ave, dim=1, descending=True)
    _, index_g2q_ave = torch.sort(score_ave, dim=0, descending=True)
    r1 = torch.sum(gallery_label == query_label_ave[index_g2q_ave[
        0, :]]).float() / len(gallery_label)
    ap50 = torch.mean(
        (gallery_label[index_q2g_ave[:, :50]] == query_label_ave.reshape(
            (-1, 1)).expand(-1, 50)).float())

    return cmc_q2g, cmc_g2q, r1, ap50
