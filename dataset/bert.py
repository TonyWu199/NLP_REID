import sys
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import copy

class bert(nn.Module):#模型代码

    def __init__(self):#初始化
        super(bert, self).__init__()
        # self.bertconfig = BertConfig('bert-base-uncased-config')
        self.model_config = BertConfig.from_pretrained('bert-base-uncased')
        self.model_config.output_hidden_states = True
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.model_config)

    def forward(self, cap_ids, masks):
        with torch.no_grad():
            # output = self.model(cap_ids,
            #                     token_type_ids=None,
            #                     attention_mask=masks)
            vector, pooler, all_hidden_states = self.model(cap_ids, token_type_ids=None, attention_mask=masks)
        # print(vector[:,0,:]==all_hidden_states[-1][:,0,:])
        text_embeddings = all_hidden_states[-4][:,0,:]
        text_embeddings = torch.cat((text_embeddings, all_hidden_states[-3][:,0,:]), 1)
        text_embeddings = torch.cat((text_embeddings, all_hidden_states[-2][:,0,:]), 1)
        text_embeddings = torch.cat((text_embeddings, all_hidden_states[-1][:,0,:]), 1)

        # text_embeddings = vector[:, 0, :]
        #output[0](batch size, sequence length, model hidden dimension)

        return text_embeddings

# transfer str -> bert input
def str2tokens(str, max_length, tokenizer):
    str_ids = [tokenizer.encode(str, add_special_tokens=True, max_length=max_length, truncation=True)]
    words_tokens = pad_sequences(str_ids, maxlen=max_length, dtype='long', value=0, truncating='post', padding='post')
    att_mask = [[int(words_token > 0) for words_token in words_tokens[0]]]

    return words_tokens, att_mask

def tokens2vec(tokens, att_mask, model, device=torch.device('cpu')):
    tokens = torch.tensor(tokens)
    tokens = tokens.to(device)
    att_mask = torch.tensor(att_mask)
    att_mask = att_mask.to(device)

    tokens = torch.tensor(tokens)
    att_mask = torch.tensor(att_mask).clone().detach()
    words_vectors = model(tokens, att_mask).to(device)
    words_vectors = words_vectors.cpu()
    return words_vectors

'''
    transfer one sentence to words using NLTK
'''
def sent2words(sentence, del_stopwords=True):
    # Normalization
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z0-9]', " ", sentence)
    # Tokenization
    words = word_tokenize(sentence)

    my_stopwords = stopwords.words('english')
    # my_stopwords.extend(['pedestrian', 'person', 'indivisual', 'female', 'male', 'woman', 'man'])
    # Stop word
    # 去掉停用词后还有7074个word
    if del_stopwords:
        words = [w for w in words if w not in my_stopwords]
    return words

if __name__ == '__main__':
    ss = []
    ss.append('The woman is wearing a weird gray hat, a blue shirt that says \"ciao bella\" and has blue colored floral pants on.')
    ss.append('The woman is walking while looking over her left shoulder.  She is wearing a white jacket and black pants.  She is carrying a white bag.')
    # ss.append('The man is wearing a grey shirt and Gree sneakers and Gree shirts.')
    ss.append('A young man with an orange t-shirt and black jeans is walking while wearing a black backpack and white sneakers.')

    # print(sent2words(ss[0]))

    bert = bert()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    

    # word_pieces = tokenizer.tokenize(ss[0])
    # print(word_pieces)

    tokens, mask = str2tokens(ss[0], 56, tokenizer)
    words_vectors = tokens2vec(tokens, mask, bert)

    # [1,768]
    # print(words_vectors)
    

    # # ------ Bert Preprocess ------
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # for s in ss:
    #     # sentence -> ids
    #     # [[101,......,102]]
    #     s_ids = [tokenizer.encode(s, add_special_tokens=True, max_length=120, truncation=True)]
    #     # 截断或补齐(value=0)
    #     s_ids = pad_sequences(s_ids, maxlen=120, dtype='long', value=0, truncating='post', padding='post')
    #     # 创建mask
    #     att_mask = [[int(s_id > 0) for s_id in s_ids[0]]]
        
    #     s_ids = torch.tensor(s_ids)
    #     att_mask = torch.tensor(att_mask)
    #     output = bert(s_ids, att_mask)

    #     result.append(output)
    # print(result[0])
    # print(torch.cosine_similarity(result[0], result[1], dim=1),\
    #         torch.cosine_similarity(result[0], result[2], dim=1))