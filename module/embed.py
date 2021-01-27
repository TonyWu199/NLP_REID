import sys,os
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import make_loss_evaluator
from utils.utils import weights_init_kaiming

class SimpleHead(nn.Module):
    def __init__(self,
                 visual_size,
                 textual_size,
                 cfg,
                 ):
        super(SimpleHead, self).__init__()
        self.cfg = cfg
        self.embed_size = self.cfg.MODEL.EMBEDDING_SIZE

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)

        # self.visual_local_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.visual_local_embed_list = nn.ModuleList([nn.Linear(visual_size, self.embed_size) for i in range(cfg.MODEL.VISUAL_MODEL.NUM_STRIPES)])
        # self.textual_local_embed_layer = nn.Linear(textual_size, self.embed_size)

        if cfg.MODEL.BN_LAYER:
            #! global
            self.bottelneck_global_visual = nn.BatchNorm1d(self.embed_size)
            self.bottelneck_global_visual.bias.requires_grad_(False)
            self.bottelneck_global_visual.apply(weights_init_kaiming)
            self.bottelneck_global_textual = nn.BatchNorm1d(self.embed_size)
            self.bottelneck_global_textual.bias.requires_grad_(False)
            self.bottelneck_global_textual.apply(weights_init_kaiming)

            #! local
            # self.bottelneck_local_visual = nn.BatchNorm1d(self.cfg.MODEL.VISUAL_MODEL.NUM_STRIPES)
            # self.bottelneck_local_visual.bias.requires_grad_(False)
            # self.bottelneck_local_visual.apply(weights_init_kaiming)

            # self.bottelneck_local_textual = nn.BatchNorm1d(self.cfg.MODEL.TEXTUAL_MODEL.MAX_LENGTH)
            # self.bottelneck_local_textual.bias.requires_grad_(False)
            # self.bottelneck_local_textual.apply(weights_init_kaiming)
            self.bottelneck_local_textual = nn.LayerNorm(self.embed_size)

        self.loss_evaluator = make_loss_evaluator(self.cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    '''
    lr in embedding is 10 times the lr of visual and textual models
    '''
    def forward(self,
                global_visual_feature,
                global_textual_feature,
                labels,
                local_visual_feat=None,
                local_textual_feat=None,
                text_length=None):
        # resnet 模型
        # if 'resnet' in self.cfg.MODEL.VISUAL.MODEL_NAME:
        #! global Embedding
        batch_size = global_visual_feature.size(0)
        global_visual_feature = self.avgpool(global_visual_feature)

        global_visual_embed = global_visual_feature.view(batch_size, -1)
        global_textual_embed = global_textual_feature.view(batch_size, -1)
        
        global_visual_embed = self.visual_embed_layer(global_visual_embed)
        global_textual_embed = self.textual_embed_layer(global_textual_embed)

        #! local Embedding
        local_visual_embed = None
        local_textual_embed = None
        if local_visual_feat != None or local_textual_feat != None:
            local_visual_embed_list = []
            for i in range(len(self.visual_local_embed_list)):
                # --> [bs, visual_size] -> [bs, embed_size]
                local_visual_embed_list.append(self.visual_local_embed_list[i](local_visual_feat[i]))
            local_visual_embed = torch.stack(local_visual_embed_list)
            local_visual_embed = local_visual_embed.permute(1,0,2)

            # ! 所有word embedding共用一个Linear层是不合理的，会导致Loss无法下降，恒定在25.600
            # ! 当word为1时，也会出现类似的问题。
            # local_textual_feat_view = local_textual_feat.view(local_textual_feat.size(0)*local_textual_feat.size(1), -1)
            # local_textual_embed = self.textual_local_embed_layer(local_textual_feat_view)


        if self.cfg.MODEL.BN_LAYER:
            #! global BN
            global_visual_embed = self.bottelneck_global_visual(global_visual_embed)
            global_textual_embed = self.bottelneck_global_textual(global_textual_embed)

            # #! local BN
            # if local_visual_feat != None or local_textual_feat != None:
            #     local_textual_embed = self.bottelneck_local_textual(local_textual_embed)
            #     local_visual_embed = self.bottelneck_local_visual(local_visual_embed)
            #     local_textual_embed = self.bottelneck_local_textual(local_textual_embed)
 

        if self.training:
            losses, precs = self.loss_evaluator(
                global_visual_embed, global_textual_embed, 
                labels,
                local_visual_embed, local_textual_embed, text_length
            )
            return None, losses, precs

        outputs = list()
        outputs.append(global_visual_embed)
        outputs.append(global_textual_embed)
        return outputs, None

        # PCB 分块
        # elif 'PCB' in self.cfg.MODEL.VISUAL.MODEL_NAME:
        #     global_visual_embed = visual_feature[0]
        #     local_visual_logits_list = visual_feature[1]

        #     batch_size = textual_feature.size(0)

        #     #! Global
        #     # global_visual_embed = self.avgpool(global_visual_embed)
        #     global_visual_embed = global_visual_embed.view(batch_size, -1)
        #     textual_embed = textual_feature.view(batch_size, -1)
            
        #     global_visual_embed = self.visual_embed_layer(global_visual_embed)
        #     textual_embed = self.textual_embed_layer(textual_embed)

        #     if self.cfg.MODEL.BN_LAYER:
        #         visual_embed = self.bottelneck_visual(global_visual_embed)
        #         textual_embed = self.bottelneck_textual(textual_embed)

        #     if self.training:
        #         losses = self.loss_evaluator(
        #             global_visual_embed, textual_embed, labels, local_visual_logits_list
        #         )
        #         return None, losses

        #     #! Local


def build_embed(visual_out_channels,
                textual_out_channels,
                cfg
            ):

    return SimpleHead(visual_out_channels,
                      textual_out_channels,
                      cfg)