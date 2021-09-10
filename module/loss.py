from module.utils import CrossEntropyLabelSmooth
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

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = torch.mm(e, e.t())
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
	
class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super(HardDarkRank, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss

class KLdivergence(nn.Module):
    
    def __init__(self, T):
        super(KLdivergence,self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.T = T
        
    def forward(self, student, teacher):
         n, c = student.size()
         assert(student.size() == teacher.size())
         # Do not BP to teacher model
         teacher = teacher.detach()
         
         student = self.softmax(student/self.T)
         teacher = self.softmax(teacher/self.T)
         
         log_student = student.clamp(min=1e-12).log()
         log_teacher = teacher.clamp(min=1e-12).log()
         
         loss = (log_teacher - log_student) * teacher
         loss = loss.sum(dim=1, keepdim=False).mean()
         
         return loss

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
    def __init__(self, cfg, num_classes, feature_size, dropout_prob=None):
        super(LossComputation, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dropout_prob = dropout_prob
        self.scale = 28
        self.KL = KLdivergence(T=self.cfg.MODEL.LOSS.T)
        self.softceloss = CrossEntropyLabelSmooth(num_classes=self.num_classes, epsilon=0.1, use_gpu=True)
        self.HardDarkRank = HardDarkRank(alpha=3, beta=3, permute_len=63)

        if self.cfg.MODEL.LOSS.INSTANCE:
            self.W = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)
            nn.init.xavier_uniform_(self.W.data, gain=1)
            self.W_text = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)
            nn.init.xavier_uniform_(self.W_text.data, gain=1)

        if self.cfg.MODEL.LOSS.CMPC:
            self.W2 = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)
            nn.init.xavier_uniform_(self.W2.data, gain=1)

        self.epsilon = 1e-8

    def instance_loss(self, visual_embed, textual_embed, labels, local_visual_embed, local_textual_embed):
        # visual_embed = local_visual_embed.squeeze(1)
        # textual_embed = local_textual_embed.squeeze(1)
        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)

        #* instance loss
        W_norm = F.normalize(self.W, p=2, dim=0)
        W_text_norm = F.normalize(self.W_text, p=2, dim=0)
        # W1_norm = F.normalize(self.W1, p=2, dim=0)

        visual_logits = self.scale * torch.matmul(visual_norm, W_norm)
        textual_logits = self.scale * torch.matmul(textual_norm, W_norm)

        # visual_logits = self.share_classifier(visual_norm)
        # textual_logits = self.share_classifier(textual_norm)

        criterion = nn.CrossEntropyLoss(reduction='mean')
        # v_loss = criterion(input=visual_logits, target=labels)
        # t_loss = criterion(input=textual_logits, target=labels)
        v_loss = self.softceloss(inputs=visual_logits, targets=labels)
        t_loss = self.softceloss(inputs=textual_logits, targets=labels)

        t_filter_loss = torch.tensor(0.0)
        distillation_loss = torch.tensor(0.0)

        #* CMKT
        # student = txt
        if self.cfg.MODEL.LOSS.LAMBDA2 != 0 or self.cfg.MODEL.LOSS.LAMBDA3 != 0:
            # feature
            visual_f = visual_norm.detach()
            distillation_loss_f = torch.pow(visual_f - textual_norm, 2).sum(dim=1, keepdim=False).mean()
            # distillation_loss_f = self.KL(textual_norm, visual_norm)

            # probability
            distillation_loss_p = self.KL(textual_logits, visual_logits)
            distillation_loss = self.cfg.MODEL.LOSS.LAMBDA2 * distillation_loss_f + self.cfg.MODEL.LOSS.LAMBDA3 * distillation_loss_p
        #* KR
        if self.cfg.MODEL.LOSS.LAMBDA4 != 0.0:
            # textual_logits_filter = self.text_classifier(textual_norm)
            textual_logits_filter = self.scale * torch.matmul(textual_norm, W_text_norm)
            # t_filter_loss = self.cfg.MODEL.LOSS.LAMBDA4 * criterion(input=textual_logits_filter, target=labels)
            t_filter_loss = self.cfg.MODEL.LOSS.LAMBDA4 * self.softceloss(inputs=textual_logits_filter, targets=labels)

        loss = v_loss + t_loss + t_filter_loss + distillation_loss

        if self.cfg.DATASET.NAME == 'Flickr30K':
            visual_norm = visual_norm.detach()
            loss_hdr = self.HardDarkRank(textual_norm, visual_norm)
            # textual_f = textual_norm.detach()
            # loss_hdr = self.HardDarkRank(visual_norm, textual_f)
            loss = loss + 0.1 * loss_hdr

        # precision calculate
        visual_pred = torch.argmax(visual_logits, dim=1)
        textual_pred = torch.argmax(textual_logits, dim=1)
        visual_precision = torch.mean((visual_pred == labels).float())
        textual_precision = torch.mean((textual_pred == labels).float())

        return loss, distillation_loss, visual_precision, textual_precision

    def global_align_loss(self, visual_embed, textual_embed, labels):
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40

        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)

        batch_size = visual_embed.size(0)
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
    @staticmethod
    def MH_LOSS(cfg, 
                visual_embed, textual_embed, 
                labels, 
                local_visual_embed=None, local_textual_embed=None, text_length=None):
        alpha = 0.2
        max_violation = True

        # feature normalization
        visual_norm = F.normalize(visual_embed, p=2, dim=-1)
        textual_norm = F.normalize(textual_embed, p=2, dim=-1)
        
        if local_visual_embed != None or local_textual_embed != None:
            local_visual_norm = F.normalize(local_visual_embed, p=2, dim=-1)
            local_textual_norm = F.normalize(local_textual_embed, p=2, dim=-1)


        if cfg.MODEL.GRAN == 'coarse':
            scores = visual_norm.mm(textual_norm.t())
        elif cfg.MODEL.GRAN == 'fine':
            # get scores
            # [visual, textual]
            global_scores = visual_norm.mm(textual_norm.t())

            # [visual, textual]
            local_scores_t2i = scores_t2i(local_visual_norm, local_textual_norm, text_length)
            # [visual, textual]
            local_scores_i2t = scores_i2t(local_visual_norm, local_textual_norm, text_length)
            local_scores = (local_scores_t2i + local_scores_i2t) / 2
            scores = global_scores + local_scores.t()*0.5
        
        # calculate loss
        diagonal = scores.diag().view(visual_embed.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (alpha + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (alpha + scores - d2).clamp(min=0)

        n = labels.size(0)
        labels_ = labels.expand(n,n)
        mask = (labels_.eq(labels_.t()))

        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if max_violation:
            cost_s = cost_s.max(dim=1)[0]
            cost_im = cost_im.max(dim=0)[0]

        # if cfg.MODEL.LOSS.LAMBDA2 == 0.0:
        #     return cost_s.sum()+cost_im.sum()
        # else:
        #     # only image retrieval
        #     return cost_im.sum()

        return cost_im.sum()

    @staticmethod
    def ranking_loss(cfg, visual_embed, textual_embed, labels):
        # max(0, -y(x1-x2) + margin)
        rank_loss = nn.MarginRankingLoss(margin=0.3)
        
        n = visual_embed.size(0)
        visual_pow = torch.pow(visual_embed, 2).sum(dim=1, keepdim=True).expand(n, n)
        textual_pow = torch.pow(textual_embed, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = visual_pow + textual_pow.t()
        dist.addmm_(visual_embed, textual_embed.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # image2text
        mask = labels.expand(n,n).eq(labels.expand(n,n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        rank_loss1 = rank_loss(dist_an, dist_ap, y)

        #text2image
        dist = dist.t()
        mask = labels.expand(n,n).eq(labels.expand(n,n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        rank_loss2 = rank_loss(dist_an, dist_ap, y)

        return rank_loss1 + rank_loss2

    # @staticmethod
    def aug_loss(self, cfg,
                visual_embed, txt1_embed, txt2_embed, aug_model, labels):
        txt1_norm = F.normalize(txt1_embed, p=2, dim=1)
        txt2_norm = F.normalize(txt2_embed, p=2, dim=1)

        W_text_norm = F.normalize(self.W_text, p=2, dim=0)
        textual_logits_filter = self.scale * torch.matmul(txt2_norm, W_text_norm)
        loss = self.softceloss(inputs=textual_logits_filter, targets=labels)

        return loss
        # return F.pairwise_distance(txt1_norm, txt2_norm, p=2).mean()

        # #* model choice          
        # if cfg.MODEL.AUG.AUGMODEL == 'VAE':
        #     g_txt1, mu1, logvar1 = aug_model(txt1_norm)    
        # elif cfg.MODEL.AUG.AUGMODEL == 'AE':
        #     encode, g_txt1 = aug_model(txt1_norm)
        # txt1_vae_norm = F.normalize(g_txt1, p=2, dim=1)

        # #* target
        # if cfg.MODEL.AUG.TYPE == 'type1':
        #     target_norm = txt2_norm
        # # elif cfg.MODEL.AUG.TYPE == 'type2':
        # #     target_feature = txt1_embed + txt2_embed
        # #     target_norm = F.normalize(target_feature, p=2, dim=1)
        # # elif cfg.MODEL.AUG.TYPE == 'type3':
        # #     target_feature = txt_feature_extractor(model, forward_txt12, forward_txt12_len)
        # #     target_norm = F.normalize(target_feature, p=2, dim=1)
        # # elif cfg.MODEL.AUG.TYPE == 'type4':
        # #     target_feature12 = txt_feature_extractor(model, forward_txt12, forward_txt12_len)
        # #     target_feature21 = txt_feature_extractor(model, backward_txt21, backward_txt21_len)
        # #     target_feature = target_feature12 + target_feature21
        # #     target_norm = F.normalize(target_feature, p=2, dim=1)

        # #*||f1-f2| - |f1-vf1||+|f2-vf1|    
        # # if cfg.MODEL.AUG.LOSS == 'loss1':
        # #     # dist11 = torch.pow(txt1_norm - txt1_vae_norm, 2).sum(dim=1, keepdim=False)
        # #     # dist12 = torch.pow(txt1_norm - txt2_norm, 2).sum(dim=1, keepdim=False)
        # #     # dist2 = torch.pow(txt1_vae_norm - txt2_norm, 2).sum(dim=1, keepdim=False).mean()
        # #     dist11 = F.pairwise_distance(txt1_norm, txt1_vae_norm, p=2, keepdim=False)
        # #     dist12 = F.pairwise_distance(txt1_norm, target_norm, p=2, keepdim=False)
        # #     dist1 = torch.abs(dist11 - dist12).mean()
        # #     dist2 = F.pairwise_distance(txt1_vae_norm, target_norm, p=2).mean()
        # #     aug_loss = dist1 + dist2 
        # #*|f2-vf1| 
        # if cfg.MODEL.AUG.LOSS == 'loss2':
        #     aug_loss = F.pairwise_distance(txt1_vae_norm, target_norm, p=2).mean()

        # return aug_loss

    def pcb_loss(self, local_visual_logits_list, labels):
        loss = torch.tensor(0.0).cuda()
        criterion = nn.CrossEntropyLoss()
        for logits in local_visual_logits_list:
            loss += criterion(logits, labels)
        return loss

    def forward(self, 
                visual_embed, textual_embed, 
                labels, 
                local_visual_embed=None, local_textual_embed=None, text_length=None,
                txt2_embed=None, aug_model=None):
        instance_loss = torch.tensor(0.0)
        distill_loss = torch.tensor(0.0)
        global_align_loss = torch.tensor(0.0)
        cmpc_loss = torch.tensor(0.0)
        cmpm_loss = torch.tensor(0.0)
        mh_loss = torch.tensor(0.0)
        sum_pcb_loss = torch.tensor(0.0)
        aug_loss = torch.tensor(0.0)
        ranking_loss = torch.tensor(0.0)
        visual_precision = torch.tensor(0.0)
        textual_precision = torch.tensor(0.0)

       
        if self.cfg.MODEL.AUG.LAMBDA5 != 0.0:
            aug_loss = self.aug_loss(self.cfg, visual_embed, textual_embed, txt2_embed, aug_model, labels=labels)
        if self.cfg.MODEL.LOSS.INSTANCE:
            instance_loss, distill_loss, visual_precision, textual_precision = self.instance_loss(visual_embed, textual_embed, labels, local_visual_embed, local_textual_embed)
        if self.cfg.MODEL.LOSS.GLOBALALIGN:
            global_align_loss = self.global_align_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.CMPC:
            cmpc_loss, visual_precision, textual_precision = self.cmpc_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.cmpm_loss(visual_embed, textual_embed, labels)
        if self.cfg.MODEL.LOSS.MH:
            mh_loss = self.MH_LOSS(self.cfg, visual_embed, textual_embed, labels, local_visual_embed, local_textual_embed, text_length)
        # if self.cfg.MODEL.VISUAL_MODEL.NAME == 'PCB':
        #     sum_pcb_loss = self.pcb_loss(local_visual_logits_list, labels)
        if self.cfg.MODEL.LOSS.RANKING:
            ranking_loss = self.ranking_loss(self.cfg, visual_embed, textual_embed, labels)

        losses = {
            "aug_loss": aug_loss,
            "instance_loss": instance_loss * self.cfg.MODEL.LOSS.INSTANCE,
            "distill_loss": distill_loss,
            "global_align_loss": global_align_loss * self.cfg.MODEL.LOSS.GLOBALALIGN,
            "cmpc_loss": cmpc_loss * self.cfg.MODEL.LOSS.CMPC,
            "cmpm_loss": cmpm_loss * self.cfg.MODEL.LOSS.CMPM,
            "mh_loss": mh_loss * self.cfg.MODEL.LOSS.MH,
            "pcb_loss": sum_pcb_loss,
            "ranking_loss": ranking_loss,
        }
        precs = {
            "visual_prec": visual_precision,
            "textual_prec": textual_precision,
        }
        return losses, precs



def make_loss_evaluator(cfg, classnum):
    # num_classes = 11003
    feature_size = 1024
    dropout_prob = 0.0
    return LossComputation(cfg, classnum, feature_size, dropout_prob)
