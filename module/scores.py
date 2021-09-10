from typing import ContextManager
from warnings import simplefilter
import torch
import torch.nn as nn
from tqdm import tqdm

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def func_attention(query, context, smooth):
    """
    query: (n_iamges, query_len, dim)
    context: (n_images, source_len, dim)
    """
    batch_sizeq, query_len = query.size(0), query.size(1)
    batch_size, source_len = context.size(0), context.size(1)

    # [bs, dim, query_len]
    queryT = query.transpose(1,2)

    # [bs, source_len, query_len]
    attn = torch.bmm(context, queryT)

    # 'softmax':
    # attn = attn.view(batch_size * source_len, query_len)
    # attn = nn.Softmax(dim=1)(attn)
    # attn = attn.view(batch_size, source_len, query_len)
    # 'clipped_l2norm'
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    
    # -> [bs, query_len, source_len]
    attn = attn.permute(0,2,1).contiguous()
    # -> [bs*query_len, source_len]
    attn = attn.view(batch_size * query_len, source_len)
    attn = nn.Softmax(dim=1)(attn * smooth)
    attn = attn.view(batch_size, query_len, source_len)
    # -> [bs, source_len, query_len]
    attnT = attn.permute(0,2,1).contiguous()

    # -> [bs, dim, source_len]
    contextT = context.permute(0,2,1)
    weightedContext = torch.bmm(contextT, attnT)
    # -> [bs, query_len, dim]
    weightedContext = weightedContext.permute(0,2,1)

    return weightedContext, attnT

# [bs, stripes, dim]
# [bs, max_len, dim]
# [bs, 1]
def scores_t2i(local_visual_embed, local_textual_embed, text_length, is_tqdm=False):
    scores = []
    n_images = local_visual_embed.size(0)
    n_captions = local_textual_embed.size(0)
    n_regions = local_visual_embed.size(1)

    is_tqdm = False
    if is_tqdm:
        iter_lst = tqdm(range(n_captions))
    else:
        iter_lst = range(n_captions)

    # 6148 * 3074 about 10 minutes
    for i in iter_lst:
        n_word = text_length[i]
        cap_i = local_textual_embed[i, :n_word, :].unsqueeze(0).contiguous()
        # [1, max_len, dim] -> [64, max_len, dim]
        cap_i_expand = cap_i.repeat(n_images, 1, 1)

        # query attention
        weiContext, attn = func_attention(cap_i_expand, local_visual_embed, smooth=20.0)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)

        # LogSumExp
        row_sim.mul_(6.0).exp_()
        if row_sim.ndimension() == 1:
            row_sim = row_sim.unsqueeze(1)
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim) / 6.0
        
        scores.append(row_sim)
    
    # (bs, bs)
    scores = torch.cat(scores, 1)

    return scores

def scores_i2t(local_visual_embed, local_textual_embed, text_length, is_tqdm=False):
    scores = []
    n_images = local_visual_embed.size(0)
    n_captions = local_textual_embed.size(0)
    n_regions = local_visual_embed.size(1)

    is_tqdm = False
    if is_tqdm:
        iter_lst = tqdm(range(n_captions))
    else:
        iter_lst = range(n_captions)

    for i in iter_lst:
        n_word = text_length[i]
        cap_i = local_textual_embed[i, :n_word, :].unsqueeze(0).contiguous()
        # [1, max_len, dim] -> [64, max_len, dim]
        cap_i_expand = cap_i.repeat(n_images, 1, 1)

        # query attention
        # -> [bs, query_len, dim]
        weiContext, attn = func_attention(local_visual_embed, cap_i_expand, smooth=9.0)
        row_sim = cosine_similarity(local_visual_embed, weiContext, dim=2)
        
        # LogSumExp
        row_sim.mul_(6.0).exp_()
        if row_sim.ndimension() == 1:
            row_sim = row_sim.unsqueeze(1)
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim) / 6.0

        scores.append(row_sim)
    
    scores = torch.cat(scores, 1)

    return scores