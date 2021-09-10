from __future__ import print_function, division
from numpy.lib.function_base import average
import torch
from torch.autograd import Variable
import pdb
from module.utils import *
import logging
from module.scores import scores_i2t, scores_t2i
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from VAE.vae import VAE

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
            # 添加图像flip操作
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            # [batch_size, 1024]
            # 是否提取分块特征
            if cfg.MODEL.VISUAL_MODEL.NUM_STRIPES != 0:
                outputs, local = model.visual_model(input_img)
            else:
                outputs = model.visual_model(input_img)    
            outputs = model.embed_model.avgpool(outputs)  
            batch_size = outputs.size(0)
            outputs = outputs.view(batch_size, -1)
            # outputs = model.embed_model.visual_embed_layer(outputs)
            # if cfg.MODEL.BN_LAYER:
            #     outputs = model.embed_model.bottelneck_global_visual(outputs)
            outputs = model.embed_model.visual_block(outputs)

            ff.append(outputs.data.cpu().detach())
        ff = ff[0] + ff[1]
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features, gallery_label

def extract_feature_query(model, loader, cfg, aug_model=None):
    features = torch.FloatTensor()
    query_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        if not cfg.DATASET.NAME in ['Birds', 'Flowers']:
            txt, label, txtlen = data
            query_label = torch.cat((query_label, label))
            input_txt = Variable(txt.cuda())
            input_txtlen = Variable(txtlen.cuda())
            batch_size = input_txt.size(0)

            #!local
            if cfg.MODEL.TEXTUAL_MODEL.WORDS:
                outputs, local_feat = model.textual_model(input_txt, input_txtlen)
            #!global
            else:
                outputs = model.textual_model(input_txt, input_txtlen)
            outputs = outputs.view(batch_size, -1)
            outputs = model.embed_model.textual_block(outputs)
        
            outputs_norm = F.normalize(outputs, p=2, dim=1)
            # # aug model
            # if aug_model != None and cfg.MODEL.AUG.LAMBDA5 != 0:
            #     if cfg.MODEL.AUG.AUGMODEL == 'VAE':
            #         outputs_vae, mu, logvar = aug_model(outputs_norm)
            #     elif cfg.MODEL.AUG.AUGMODEL == 'AE':
            #         encode, outputs_vae = aug_model(outputs_norm)

            #     outputs = (1-cfg.MODEL.AUG.LAMBDA5)*outputs + cfg.MODEL.AUG.LAMBDA5*outputs_vae
            
            ff = outputs.data.cpu().detach()
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)

        # loader query for cub and flowers 
        else:
            # [[txt, label, txtlen],
            # [txt, label, txtlen],
            # [txt, label, txtlen]]
            # ...

            FF = []
            global_label = None # 10 sentence has same labels
            ret_lst = data
            for lst in ret_lst:
                txt, label, txtlen = lst[0]
                global_label = label
                input_txt = Variable(txt.cuda())
                input_txtlen = Variable(txtlen.cuda())
                batch_size = input_txt.size(0)
                
                outputs = model.textual_model(input_txt, input_txtlen)
                outputs = outputs.view(batch_size, -1)
                outputs = model.embed_model.textual_embed_layer(outputs)
                outputs = model.embed_model.bottelneck_global_textual(outputs)

                FF.append(outputs.data.cpu().detach())

            # 计算平均特征
            ff = FF[0] + FF[1] + FF[2] + FF[3] + FF[4] + FF[5] + FF[6] + FF[7] + FF[8] + FF[9]
            # average_feature /= len(ret_lst) 
            # total_feature = torch.zeros_like(text_feature_lst_per_img[0])
            # for feature in text_feature_lst_per_img:
            #     total_feature += feature
            # average_feature = total_feature / len(ret_lst)

            # ff = average_feature.data.cpu().detach()
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            query_label = torch.cat((query_label, global_label))
            features = torch.cat((features, ff), 0)
    return features, query_label

def get_local_score(model, loader_query, loader_gallery, cfg):
    gallery_labels = torch.tensor([], dtype=torch.int64) #3074
    visual_local_features = torch.FloatTensor()
    # 因为img的regions个数固定，首先抽取img的特征
    for gallery_data in loader_gallery:
        img, gallery_label = gallery_data
        gallery_labels = torch.cat((gallery_labels, gallery_label))
        input_img = img.cuda()

        ff = []
        _, local_feature_list = model.visual_model(input_img)
        local_visual_embed_list = []
        for i in range(len(model.embed_model.visual_local_embed_list)):
            local_visual_embed_list.append(model.embed_model.visual_local_embed_list[i](local_feature_list[i]))
        local_visual_embed = torch.stack(local_visual_embed_list).permute(1,0,2)
        ff = local_visual_embed.data.cpu()
        ff = F.normalize(ff, p=2, dim=-1)
        # [3074, num_stripes, dim]
        visual_local_features = torch.cat((visual_local_features, ff), 0)
    
    
    query_labels = torch.tensor([], dtype=torch.int64) #6148
    txt_lens = torch.tensor([], dtype=torch.int64)
    textual_local_features = torch.FloatTensor()
    for query_data in loader_query:
        txt, query_label, txt_len = query_data
        query_labels = torch.cat((query_labels, query_label))
        txt_lens = torch.cat((txt_lens, txt_len))
        input_txt = txt.cuda()
        input_txtlen = txt_len.cuda()

        query_feat, local_query_feat = model.textual_model(input_txt, input_txtlen)
        ff = local_query_feat.data.cpu()
        ff = F.normalize(ff, p=2, dim=-1)

        ff.resize_((ff.size(0), cfg.MODEL.TEXTUAL_MODEL.MAX_LENGTH, ff.size(2)))

        textual_local_features = torch.cat((textual_local_features, ff), 0)

    total_score = scores_t2i(visual_local_features, textual_local_features, txt_lens, is_tqdm=True)
    total_score += scores_i2t(visual_local_features, textual_local_features, txt_lens, is_tqdm=True)
    total_score = total_score / 2
    # [3074, 6148] -> [6148, 3074]
    total_score = total_score.t()
    return total_score, query_labels, gallery_labels

# def extract_localfeature_gallery(model, loader, cfg):
#     pass
# def extract_localfeature_query(model, loader, cfg):
#     features = torch.FloatTensor()
#     query_label = torch.tensor([], dtype=torch.int64)
#     for data in loader:
#         txt, label, txtlen = data
#         query_label = torch.cat((query_label, label))
#         input_txt = Variable(txt.cuda())
#         input_txtlen = Variable(txtlen.cuda())
#         batch_size = input_txt.size(0)

#         #!local
#         if cfg.MODEL.TEXTUAL_MODEL.WORDS:
#             outputs, local_feat = model.textual_model(input_txt, input_txtlen)
#             pass
#         #!global
#         else:
#             outputs = model.textual_model(input_txt, input_txtlen)

#         outputs = outputs.view(batch_size, -1)
#         outputs = model.embed_model.textual_embed_layer(outputs)
#         if cfg.MODEL.BN_LAYER:
#             outputs = model.embed_model.bottelneck_global_textual(outputs)

#         ff = outputs.data.cpu()

#         # norm feature
#         fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
#         ff = ff.div(fnorm.expand_as(ff))

#         features = torch.cat((features, ff), 0)
#     return features, query_label

def test(model, loader_query, loader_gallery, cfg, avenorm=False, aug_model=None):
    if aug_model != None:
        print("Testing with {}".format(cfg.MODEL.AUG.AUGMODEL))
    if cfg.DATASET.NAME == 'Birds' or cfg.DATASET.NAME == 'Flowers':
        print("Testing with averaged features")

    # ! global score
    with torch.no_grad():
        query_feature, query_label = extract_feature_query(model, loader_query, cfg, aug_model)
        gallery_feature, gallery_label = extract_feature_gallery(model, loader_gallery, cfg)
    # print(query_feature.size())
    # [6148, 3074]
    score = torch.mm(query_feature, gallery_feature.t())
    
    if cfg.MODEL.GRAN == 'fine':
        #! local score
        local_scores, query_label, gallery_label = get_local_score(model, loader_query, loader_gallery, cfg)
        score = score + 0.5 * local_scores

    # t2i, i2t
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

    # r1, ap50
    # query_feature_ave = torch.tensor([])
    # query_label_ave = torch.tensor([], dtype=torch.int64)
    # query_feature_temp = query_feature[0, :]
    # sameidnum = 1.0
    # for i in range(len(query_label) - 1):
    #     if query_label[i + 1] == query_label[i]:
    #         query_feature_temp = query_feature_temp + query_feature[i + 1, :]
    #         sameidnum = sameidnum + 1
    #     elif query_label[i + 1] > query_label[i]:
    #         query_feature_temp = query_feature_temp / sameidnum
    #         query_feature_ave = torch.cat(
    #             (query_feature_ave, query_feature_temp.reshape(1, -1)), 0)
    #         query_label_ave = torch.cat(
    #             (query_label_ave, query_label[i].unsqueeze(0)))
    #         query_feature_temp = query_feature[i + 1, :]
    #         sameidnum = 1.0
    #     else:
    #         raise Exception('label is not ascending')
    # query_feature_temp = query_feature_temp / sameidnum
    # query_feature_ave = torch.cat(
    #     (query_feature_ave, query_feature_temp.reshape(1, -1)), 0)
    # query_label_ave = torch.cat(
    #     (query_label_ave, query_label[i + 1].unsqueeze(0)))
    # if avenorm:
    #     fnorm = torch.norm(query_feature_ave, p=2, dim=1, keepdim=True)
    #     query_feature_ave = query_feature_ave.div(
    #         fnorm.expand_as(query_feature_ave))
    # score_ave = torch.mm(query_feature_ave, gallery_feature.t())
    # _, index_q2g_ave = torch.sort(score_ave, dim=1, descending=True)
    # _, index_g2q_ave = torch.sort(score_ave, dim=0, descending=True)
    # r1 = torch.sum(gallery_label == query_label_ave[index_g2q_ave[
    #     0, :]]).float() / len(gallery_label)
    # ap50 = torch.mean(
    #     (gallery_label[index_q2g_ave[:, :50]] == query_label_ave.reshape(
    #         (-1, 1)).expand(-1, 50)).float())
    r1, ap50 = 0, 0

    return cmc_q2g, cmc_g2q, r1, ap50
