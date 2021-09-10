import torch
from torch import nn

from .backbones.visual import build_resnet
from .backbones.textual import build_textual_model
from .embed import build_embed


class Network(nn.Module):

    def __init__(self, cfg, classnum):
        super(Network, self).__init__()
        self.cfg = cfg
        self.visual_model = build_resnet(cfg)
        self.textual_model = build_textual_model(cfg.MODEL.TEXTUAL_MODEL.NAME, cfg)
                
        self.embed_model = build_embed(
            self.visual_model.out_channels,
            self.textual_model.out_channels,
            cfg,
            classnum
        )

    def forward(self, images, text, text_length, labels, txt2_embed=None, aug_model=None):
        local_visual_feat = None
        local_textual_feat = None

        # 是否提取分块特征
        if self.cfg.MODEL.VISUAL_MODEL.NUM_STRIPES != 0:
            # visual_feature: [bs, visual_model.out_channels]
            # local_textual: [bs, num_stripes, visual_model.out_channels]
            global_visual_feat, local_visual_feat = self.visual_model(images)
        else:
            global_visual_feat = self.visual_model(images)

        if self.cfg.MODEL.TEXTUAL_MODEL.WORDS:
            # textual_feat: [bs, textual_model.out_channels]
            # local_textual:[bs, seq_len, embbedding_size]
            global_textual_feat, local_textual_feat = self.textual_model(text, text_length)
        else:
            global_textual_feat = self.textual_model(text, text_length)
            
        outputs_embed, losses_embed, prec = self.embed_model(
            global_visual_feat, global_textual_feat, labels, local_visual_feat, local_textual_feat, text_length,
            txt2_embed, aug_model)

        if self.training:
            losses = {}
            losses.update(losses_embed)
            precs = {}
            precs.update(prec)
            return outputs_embed, losses, precs

        return outputs_embed


def build_model():
    return Network()

if __name__ == '__main__':
    vitaa = Network()
    input_text = torch.randn(8, 56, 768)
    input_text_length = torch.LongTensor([1,2,31,4,11,2,23,14])
    input_image = torch.randn(8, 3, 224, 224)
    input_labels = torch.LongTensor([1,2,3,4,5,6,7,8])
    output = vitaa(input_image, input_text, input_text_length, input_labels)
    print(output.shape)
    
