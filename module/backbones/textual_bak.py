import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from ..self_attention import SelfAttention, get_mask

MODEL_REGISTRY = {}
            
def build_textual_model(model_name, *args, **kwargs):
    print('Getting registed textual model {}'.format(model_name))
    return MODEL_REGISTRY[model_name](*args, **kwargs)

def register_model(name):
    '''DEcorator and register a new model'''
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

# def build_lstm(cfg):
#     model = BiLSTM(cfg.MODEL.TEXTUAL_MODEL.EMBEDDING_SIZE, \
#                     cfg.MODEL.TEXTUAL_MODEL.HIDDEN_SIZE, \
#                     cfg.MODEL.TEXTUAL_MODEL.DROPOUT, \
#                     cfg.MODEL.TEXTUAL_MODEL.BIDIRECTION)
#     return model

@register_model('bigru')
class BiGRU(nn.Module):
    def __init__(self, cfg):
        super(BiGRU, self).__init__()
        self.cfg = cfg

        self.embedding_size = 768 #cfg.MODEL.TEXTUAL_MODEL.EMBEDDING_SIZE
        self.hidden_dim = 512 #cfg.MODEL.TEXTUAL_MODEL.HIDDEN_SIZE
        self.drop_out = 0 #cfg.MODEL.TEXTUAL_MODEL.DROPOUT

        self.bigru = nn.GRU(self.embedding_size, 
                            self.hidden_dim, 
                            dropout=self.drop_out,
                            num_layers=1, 
                            bidirectional=True)
        self.out_channels = self.hidden_dim*2
    
    def forward(self, text, text_length):
        batch_size = text.size(0)
        embed = text

        bigru_out = self.bigru_out(embed, text_length, 0)
        
        bigru_out_max, _ = torch.max(bigru_out, dim=1)
        # bigru_out = bigru_out.view(-1, batch_size, bigru_out.size(-1)).squeeze(0)
        if self.cfg.MODEL.TEXTUAL_MODEL.WORDS:           
            #! 使用bigru
            return bigru_out_max, bigru_out

        return bigru_out


    def bigru_out(self, embed, text_length, index):
        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()

        embed_sort = embed.index_select(0, idx_sort)
        text_length = text_length.cpu()
        length_list = text_length[idx_sort]

        pack = nn.utils.rnn.pack_padded_sequence(embed_sort, length_list, batch_first=True)
        bigru_sort_out, _ = self.bigru(pack)
        bigru_sort_out = nn.utils.rnn.pad_packed_sequence(bigru_sort_out, batch_first=True)
        bigru_sort_out = bigru_sort_out[0]

        bigru_out = bigru_sort_out.index_select(0, idx_unsort)
        return bigru_out

    def init_hidden(self):
        return Variable(torch.zeros(2, 64, self.hidden_dim))


@register_model('bilstm')
class BiLSTM(nn.Module):
    def __init__(self, cfg):
        super(BiLSTM, self).__init__()
        self.cfg = cfg

        self.embedding_size = cfg.MODEL.TEXTUAL_MODEL.EMBEDDING_SIZE
        self.hidden_dim = cfg.MODEL.TEXTUAL_MODEL.HIDDEN_SIZE
        self.drop_out = cfg.MODEL.TEXTUAL_MODEL.DROPOUT
        self.num_layers = cfg.MODEL.TEXTUAL_MODEL.NUM_LAYERS
        self.atn_layers = cfg.MODEL.TEXTUAL_MODEL.ATN_LAYERS

        self.bilstm = nn.ModuleList()
        self.bilstm.append(nn.LSTM(self.embedding_size, 
                                   self.hidden_dim, 
                                   num_layers=self.num_layers, 
                                   dropout=self.drop_out, 
                                   bidirectional=False, 
                                   bias=False))
        self.out_channels = self.hidden_dim
        self.bidirectional = cfg.MODEL.TEXTUAL_MODEL.BIDIRECTION
        if self.bidirectional:
            self.bilstm.append(nn.LSTM(self.embedding_size, 
                                       self.hidden_dim, 
                                       num_layers=self.num_layers, 
                                       dropout=self.drop_out, 
                                       bidirectional=False, 
                                       bias=False))
            self.out_channels = self.hidden_dim*2

        # self.SA = nn.Linear(self.out_channels, 1)
        # # self.SA2 = nn.Linear(int(self.out_channels / 2), 1)
        # self.softmax = nn.Softmax(dim=1)
        # self.tanh = nn.Tanh()

        self._init_weight()
        # # for creating query
        # self.dropout_layer = nn.Dropout(0.5)

        # self.attention_layer = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.SA = SelfAttention(hidden_size=self.embedding_size,
        #                         num_attention_heads=16,
        #                         dropout_prob=0.2)

    def forward(self, text, text_length):
        batch_size = text.size(0)
        embed = text

        # unidirectional lstm
        bilstm_out, final_hidden_state = self.bilstm_out(embed, text_length, 0)
        
        if self.bidirectional:
            index_reverse = list(range(embed.shape[0]-1, -1, -1))
            index_reverse = torch.LongTensor(index_reverse).cuda()
            embed_reverse = embed.index_select(0, index_reverse)
            text_length_reverse = text_length.index_select(0, index_reverse)
            bilstm_out_bidirection, final_hidden_state_bidirection = self.bilstm_out(embed_reverse, text_length_reverse, 1)
            bilstm_out_bidirection_reverse = bilstm_out_bidirection.index_select(0, index_reverse)
            # final_hidden_state_bidirection = final_hidden_state_bidirection.index_select(0, index_reverse)
            bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=2)
            # final_hidden_state = torch.cat([final_hidden_state, final_hidden_state_bidirection], dim=1)
    
        # apply attention
        if self.atn_layers == 'easy':
            semantic_bilstm_out = self.attention_net(bilstm_out)
        else:
            # [LSTM](https://zhuanlan.zhihu.com/p/79064602)
            # 取最常序列时刻的输出层, [bs, seq_len, dim] -> [bs, dim]
            semantic_bilstm_out, _ = torch.max(bilstm_out, dim=1)

        # bilstm_out = bilstm_out_max.unsqueeze(1)
        if self.cfg.MODEL.TEXTUAL_MODEL.WORDS:           
            # region ! 使用self-attention
            # mask = get_mask(text_length.unsqueeze(1), self.cfg.MODEL.TEXTUAL_MODEL.MAX_LENGTH)
            # words_embedding = self.SA(text, mask)
            # mask_words_embedding = words_embedding * mask.unsqueeze(2)
            # return bilstm_out_max, mask_words_embedding
            # endregion

            #! 使用LSTM中的hidden state
            return semantic_bilstm_out, bilstm_out

        return semantic_bilstm_out

    def bilstm_out(self, embed, text_length, index):
        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)#text_length[k] is the first (idx_unsort[k]) max

        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()

        embed_sort = embed.index_select(0, idx_sort)
        text_length = text_length.cpu()
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(embed_sort, length_list, batch_first=True)

        # Type(bilstm_sort_out): PackedSequence
        # final_hidden_state: (num_layers, batch_size, hidden_dim)
        bilstm_sort_out, (final_hidden_state, final_cell_state) = self.bilstm[index](pack)
        
        bilstm_sort_out = nn.utils.rnn.pad_packed_sequence(bilstm_sort_out, batch_first=True)
        # bilstm_sort_out : (embed, seq_length)
        # embed: [batch_size, seq_length, hidden_dim]
        bilstm_sort_out = bilstm_sort_out[0]
        bilstm_out = bilstm_sort_out.index_select(0, idx_unsort)

        final_hidden_state = final_hidden_state.permute(1,0,2)
        final_hidden_state = final_hidden_state.index_select(0, idx_unsort)

        # bilstm_out: [batch_size, seq_length, hidden_dim]
        # final_hiddent_state: [batch_size, num_layers*num_directions, hidden_dim]
        return bilstm_out, final_hidden_state

    # https://www.icode9.com/content-4-681142.html
    def attention_net(self, x):
        # tan_x = self.tanh(x)
        attn = self.SA(x)
        attn = self.tanh(attn)
        # attn = self.SA2(attn)
        soft_attn = self.softmax(attn)

        weight_x = soft_attn * x
        bilstm_out_sum = torch.sum(weight_x, dim=1)
        return bilstm_out_sum

    # # https://www.cnblogs.com/douzujun/p/13511237.html
    # def attention_net(self, x, query, mask=None):
    #     # 获取 query 维度
    #     d_k = query.size(-1)

    #     scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
    #     alpha_n = F.softmax(scores, dim=-1)

    #     context = torch.matmul(alpha_n, x).sum(1)

    #     return context, alpha_n

    # # https://blog.csdn.net/dendi_hust/article/details/94435919
    # def attention_net_with_w(self, lstm_out, lstm_hidden):
    #     # [batch_size, seq_length, hidden_dim*2] -> [batch_size, seq_length, hidden_dim]
    #     lstm_tmp = torch.chunk(lstm_out, 2, -1)
    #     h = lstm_tmp[0] + lstm_tmp[1]

    #     # [batch_size, num_layers * num_directions, hidden_dim] -> [batch_size, 1, hidden_dim]
    #     lstm_hidden = torch.sum(lstm_hidden, dim=1)
    #     lstm_hidden = lstm_hidden.unsqueeze(1)
    #     atten_w = self.attention_layer(lstm_hidden)

    #     # m [batch_size, seq_len, hidden_dim]
    #     m = nn.Tanh()(h)

    #     # bmm: a(z,x,y) b(z,y,c) -> c(z,x,c)
    #     # atn_context: [batch_size, 1, seq_len] 
    #     atn_context = torch.bmm(atten_w, m.transpose(1, 2))
    #     softmax_w = F.softmax(atn_context, dim=-1)
    #     context = torch.bmm(softmax_w, h)
    #     result = context.squeeze(1)

    #     return result


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # lstm = build_lstm(bidirectional=True)
    # input1 = torch.randn(8, 56, 768)
    # input2 = Variable(torch.LongTensor([1,2,31,4,11,2,23,14]))
    # output = lstm(input1, input2)
    # print(output.shape)
    bigru = build_textual_model('bigru', cfg=None)
    input1 = torch.randn(8,56,768)
    input2 = Variable(torch.LongTensor([1,2,31,4,11,2,23,14]))
    output = bigru(input1, input2)
    print(output)
