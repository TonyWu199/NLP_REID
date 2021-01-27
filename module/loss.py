import sys,os
sys.path.append('..')

import math
from numpy.lib.twodim_base import diag
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.autograd import Variable
from utils.utils import weights_init_kaiming

from .self_attention import get_mask
from .scores import scores_i2t, scores_t2i

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class LossComputation(nn.Module):
    def __init__(self, cfg, num_classes, feature_size, dropout_prob):
        super(LossComputation, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dropout_prob = dropout_prob
        self.scale = 28
        self.margin = 0.2
        self.max_violation = True

        if self.cfg.MODEL.LOSS.INSTANCE:
            self.W = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)
            nn.init.xavier_uniform_(self.W.data, gain=1)
            # self.share_classifier = nn.Linear(self.feature_size, self.num_classes, True)
            # self.share_classifier.apply(weights_init_kaiming)

        if self.cfg.MODEL.LOSS.CMPC:
            self.W2 = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)
            nn.init.xavier_uniform_(self.W2.data, gain=1)

        self.epsilon = 1e-8

    def instance_loss(self, visual_embed, textual_embed, labels):
        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)

        W_norm = F.normalize(self.W, p=2, dim=0)
        visual_logits = self.scale * torch.matmul(visual_norm, W_norm)
        textual_logits = self.scale * torch.matmul(textual_norm, W_norm)
        # visual_logits = self.share_classifier(visual_norm)
        # textual_logits = self.share_classifier(textual_norm)

        criterion = nn.CrossEntropyLoss(reduction='mean')
        v_loss = criterion(input=visual_logits, target=labels)
        t_loss = criterion(input=textual_logits, target=labels)
        
        # 使用PCB时，全局特征的logits不进行损失计算
        # if 'PCB' in self.cfg.MODEL.VISUAL_MODEL.NAME:
        #     # v_loss = torch.tensor(0.0)
        #     loss = t_loss
        # else:
            # loss = v_loss + t_loss
        loss = v_loss + t_loss

        visual_pred = torch.argmax(visual_logits, dim=1)
        textual_pred = torch.argmax(textual_logits, dim=1)

        visual_precision = torch.mean((visual_pred == labels).float())
        textual_precision = torch.mean((textual_pred == labels).float())

        return loss, visual_precision, textual_precision

    def global_align_loss(self, visual_embed, textual_embed, labels):
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40

        batch_size = visual_embed.size(0)
        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)
        similarity = torch.matmul(visual_norm, textual_norm.t())
        mask = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t())

        loss = 0
        for i in range(batch_size):
            pred = similarity[i]
            label = mask[i].float()
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
            loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
            loss += loss_pos.sum() + loss_neg.sum()

            pred = similarity[:, i]
            label = mask[:, i].float()
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
            loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
            loss += loss_pos.sum() + loss_neg.sum()

        loss /= batch_size
        return loss

    def cmpc_loss(self, visual_embed, textual_embed, labels):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W2_norm = F.normalize(self.W2, p=2, dim=0).cuda()

        # 特征维度求2范数
        visual_norm = visual_embed / visual_embed.norm(dim=1, keepdim=True)
        textual_norm = textual_embed / textual_embed.norm(dim=1, keepdim=True)

        visual_proj_textual = torch.sum(visual_embed * textual_norm, dim=1, keepdim=True) * textual_norm
        textual_proj_visual = torch.sum(textual_embed * visual_norm, dim=1, keepdim=True) * visual_norm

        visual_logits = torch.matmul(visual_proj_textual, self.W2_norm)
        textual_logits = torch.matmul(textual_proj_visual, self.W2_norm)

        cmpc_loss = criterion(visual_logits, labels) + criterion(textual_logits, labels)

        visual_pred = torch.argmax(visual_logits, dim=1)
        textual_pred = torch.argmax(textual_logits, dim=1)

        visual_precision = torch.mean((visual_pred == labels).float())
        textual_precision = torch.mean((textual_pred == labels).float())

        return cmpc_loss, visual_precision, textual_precision

    def cmpm_loss(self, visual_embed, textual_embed, labels):
        batch_size = visual_embed.size(0)

        labels_reshape = torch.reshape(labels, (batch_size, 1))
        # 自动广播
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        # 特征维度求2范数
        visual_norm = visual_embed / visual_embed.norm(dim=1, keepdim=True)
        textual_norm = textual_embed / textual_embed.norm(dim=1, keepdim=True)
        visual_proj_textual = torch.matmul(visual_embed, textual_norm.t())
        textual_proj_visual = torch.matmul(textual_embed, visual_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(visual_proj_textual, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(visual_proj_textual, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        t2i_pred = F.softmax(textual_proj_visual, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(textual_proj_visual, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(visual_norm, textual_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    '''
        Max Hinge Loss from 
        Faghri F , Fleet D J , Kiros J R , et al. VSE++: Improving Visual-Semantic Embeddings with Hard Negatives[J]. arXiv, 2017. 
    '''
    def MH_LOSS(self, 
                visual_embed, textual_embed, 
                labels, 
                local_visual_embed=None, local_textual_embed=None, text_length=None):
        # 没有局部特征，则直接利用全局特征计算相似度
        if local_visual_embed == None or local_textual_embed == None:
            visual_norm = F.normalize(visual_embed, p=2, dim=-1)
            textual_norm = F.normalize(textual_embed, p=2, dim=-1)

            scores = visual_norm.mm(textual_norm.t())
        # 有局部特征，利用局部特征计算相似度
        else:
            local_visual_norm = F.normalize(local_visual_embed, p=2, dim=-1)
            local_textual_norm = F.normalize(local_textual_embed, p=2, dim=-1)

            scores = scores_t2i(local_visual_norm, local_textual_norm, text_length)
            # local_visual_norm = local_visual_norm.squeeze(1)
            # local_textual_norm = local_textual_norm.squeeze(1)

            # scores = local_visual_norm.mm(local_textual_norm.t())

            # mask = get_mask(text_length.unsqueeze(1), self.cfg.MODEL.TEXTUAL_MODEL.MAX_LENGTH).unsqueeze(2).expand_as(local_textual_norm_nomask)
            # # mask = mask.unsqueeze(2)
            # # mask = mask.expand_as(local_textual_norm)
            # local_textual_norm = local_textual_norm_nomask * mask
            
            # #! ImageRegion-Related-Textual
            # # [bs, stripes, dim] @ [bs, dim, max_length] => [bs, stripes, max_length]
            # attn_i2t = local_visual_norm.bmm(local_textual_norm.permute(0,2,1))
            # attn_i2t_norm = F.softmax(attn_i2t, dim=2)
            # # [bs, stripes, max_length, 1] * [bs, 1, max_length, dim]
            # attn_local_textual_norm = attn_i2t_norm.unsqueeze(3) * local_textual_norm.unsqueeze(1)
            # # [bs, stripes, dim]
            # IRRelated_textual = torch.sum(attn_local_textual_norm, dim=2)
            # # [bs, bs, stripes, stripes]
            # score_i2t = torch.einsum('abc,dec -> adbe', IRRelated_textual, local_visual_norm)
            # score_i2t_eye = torch.einsum('...ii -> ...i', score_i2t)
            # score_i2t_mean = torch.mean(score_i2t_eye, dim=2)

            # #! Word-Related-Visual
            # # [bs, max_length, dim] @ [bs, dim, stripes] => [bs, max_length, stripes]
            # attn_t2i = local_textual_norm.bmm(local_visual_norm.permute(0,2,1))
            # attn_t2i_norm = F.softmax(attn_t2i, dim=2)
            # # [bs, max_length, stripes, 1] * [bs, 1, stripes, dim]
            # attn_local_visual_norm = attn_t2i_norm.unsqueeze(3) * local_visual_norm.unsqueeze(1)
            # # [bs, max_length, dim]
            # WRelated_visual = torch.sum(attn_local_visual_norm, dim=2)
            # # [bs, bs, max_length, max_length]
            # score_t2i = torch.einsum('abc,dec -> adbe', WRelated_visual, local_textual_norm)
            # score_t2i_eye = torch.einsum('...ii -> ...i', score_t2i)
            # score_t2i_mean = torch.mean(score_t2i_eye, dim=2)
            
            # final simialrity scores
            # scores = (score_i2t_mean + score_t2i_mean) / 2
            # scores = score_t2i_mean

        
        diagonal = scores.diag().view(visual_embed.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        n = labels.size(0)
        labels_ = labels.expand(n,n)
        mask = (labels_.eq(labels_.t()))

        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # if args.max_violation:
        if self.max_violation:
            cost_s = cost_s.max(dim=1)[0]
            cost_im = cost_im.max(dim=0)[0]

        return cost_s.sum()+cost_im.sum()

    def pcb_loss(self, local_visual_logits_list, labels):
        loss = torch.tensor(0.0).cuda()
        criterion = nn.CrossEntropyLoss()
        for logits in local_visual_logits_list:
            loss += criterion(logits, labels)
        return loss

    def forward(self, 
                visual_embed, textual_embed, 
                labels, 
                local_visual_embed=None, local_textual_embed=None, text_length=None):
        instance_loss = torch.tensor(0.0)
        global_align_loss = torch.tensor(0.0)
        cmpc_loss = torch.tensor(0.0)
        cmpm_loss = torch.tensor(0.0)
        mh_loss = torch.tensor(0.0)
        sum_pcb_loss = torch.tensor(0.0)
        visual_precision = torch.tensor(0.0)
        textual_precision = torch.tensor(0.0)

        if self.cfg.MODEL.LOSS.INSTANCE:
            instance_loss, visual_precision, textual_precision = self.instance_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.GLOBALALIGN:
            global_align_loss = self.global_align_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.CMPC:
            cmpc_loss, visual_precision, textual_precision = self.cmpc_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.cmpm_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.MH:
            # mh_loss = (self.MH_LOSS(visual_embed, textual_embed, labels, local_visual_embed, local_textual_embed, text_length)\
            #             +self.MH_LOSS(textual_embed, visual_embed, labels, local_visual_embed, local_textual_embed, text_length)) / 2
            mh_loss = self.MH_LOSS(visual_embed, textual_embed, labels, local_visual_embed, local_textual_embed, text_length)
        # if self.cfg.MODEL.VISUAL_MODEL.NAME == 'PCB':
        #     sum_pcb_loss = self.pcb_loss(local_visual_logits_list, labels)

        losses = {
            "instance_loss": instance_loss * self.cfg.MODEL.LOSS.INSTANCE,
            "global_align_loss": global_align_loss * self.cfg.MODEL.LOSS.GLOBALALIGN,
            "cmpc_loss": cmpc_loss * self.cfg.MODEL.LOSS.CMPC,
            "cmpm_loss": cmpm_loss * self.cfg.MODEL.LOSS.CMPM,
            "mh_loss": mh_loss * self.cfg.MODEL.LOSS.MH,
            "pcb_loss": sum_pcb_loss,
        }
        precs = {
            "visual_prec": visual_precision,
            "textual_prec": textual_precision,
        }
        return losses, precs



def make_loss_evaluator(cfg):
    num_classes = 11003
    feature_size = 1024
    dropout_prob = 0.0
    return LossComputation(cfg, num_classes, feature_size, dropout_prob)
