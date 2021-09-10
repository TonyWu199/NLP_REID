import sys
from nltk.util import pr
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
from typing import Tuple

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


@register_model('bi_lstm')
class BiLSTM(nn.Module):
    def __init__(self, cfg):
        super(BiLSTM, self).__init__()
        self.cfg = cfg

        self.embedding_size = cfg.MODEL.TEXTUAL_MODEL.EMBEDDING_SIZE
        self.hidden_dim = cfg.MODEL.TEXTUAL_MODEL.HIDDEN_SIZE
        self.drop_out = cfg.MODEL.TEXTUAL_MODEL.DROPOUT
        self.num_layers = cfg.MODEL.TEXTUAL_MODEL.NUM_LAYERS
        self.atn_layers = cfg.MODEL.TEXTUAL_MODEL.ATN_LAYERS
        self.bidirectional = cfg.MODEL.TEXTUAL_MODEL.BIDIRECTION

        self.bilstm = nn.ModuleList()
        self.bilstm.append(nn.LSTM(self.embedding_size, 
                                   self.hidden_dim, 
                                   num_layers=self.num_layers, 
                                   dropout=self.drop_out, 
                                   bidirectional=False, 
                                   bias=False))
        self.out_channels = self.hidden_dim
        if self.bidirectional:
            self.bilstm.append(nn.LSTM(self.embedding_size, 
                                       self.hidden_dim, 
                                       num_layers=self.num_layers, 
                                       dropout=self.drop_out, 
                                       bidirectional=False, 
                                       bias=False))
            self.out_channels = self.hidden_dim*2

        self._init_weight()

    def forward(self, text, text_length):
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
            bilstm_out = torch.cat([bilstm_out, bilstm_out_bidirection_reverse], dim=2)
        semantic_bilstm_out, _ = torch.max(bilstm_out, dim=1)

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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


@register_model('bi_lnlstm')
class LayerNormLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = cfg.MODEL.TEXTUAL_MODEL.EMBEDDING_SIZE
        self.hidden_size = cfg.MODEL.TEXTUAL_MODEL.HIDDEN_SIZE
        self.num_layers = cfg.MODEL.TEXTUAL_MODEL.NUM_LAYERS
        self.bidirections = cfg.MODEL.TEXTUAL_MODEL.BIDIRECTION

        num_directions = 2 if self.bidirections else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(self.input_size if layer == 0 else self.hidden_size * num_directions),
                            hidden_size=self.hidden_size, bias=False)
            for layer in range(self.num_layers)
        ])
        self.out_channels = self.hidden_size
        if self.bidirections:
            self.hidden1 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(self.input_size if layer == 0 else self.hidden_size * num_directions),
                            hidden_size=self.hidden_size, bias=False)
            for layer in range(self.num_layers)
            ])
            self.out_channels = self.hidden_size*2

    def forward(self, input, text_length, hidden=None):
        input = input.permute(1,0,2)
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only
        num_directions = 2 if self.bidirections else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirections:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y  = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y  = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])

        y,_ = torch.max(y, dim=0)
        return y#y, (hy, cy)

if __name__ == '__main__':
    lstm = build_textual_model('bilstm', cfg=None)
    input1 = torch.randn(8, 56, 768)
    input2 = Variable(torch.LongTensor([1,2,31,4,11,2,23,14]))
    output = lstm(input1, input2)
    print(output.shape)
    # bigru = build_textual_model('bigru', cfg=None)
    # input1 = torch.randn(8,56,768)
    # input2 = Variable(torch.LongTensor([1,2,31,4,11,2,23,14]))
    # output = bigru(input1, input2)
    # print(output)
