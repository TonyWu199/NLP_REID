import scipy.io
import torch
import numpy as np
import time
import os

import pdb

#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)

    good_index = query_index
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    # print(len(CMC_tmp), index.shape)
    return CMC_tmp, index


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
result = scipy.io.loadmat('./evaluate/result/result_mhloss_refiner.mat')
# result = scipy.io.loadmat('./evaluate/result/result_baseline.mat')
query_feature = result['query_f']
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_label = result['gallery_label'][0]
# multi = os.path.isfile('multi_query.mat')

# if multi:
#     m_result = scipy.io.loadmat('multi_query.mat')
#     mquery_feature = m_result['mquery_f']
#     mquery_cam = m_result['mquery_cam'][0]
#     mquery_label = m_result['mquery_label'][0]
    
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
ap_all = np.zeros(len(query_label))
cmc_all = np.zeros((len(query_label),3))
index_all = np.zeros((len(query_label),len(gallery_label)),dtype=np.int)
for i in range(len(query_label)):
    tmp, index_all[i] = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
    ap_tmp, CMC_tmp = tmp
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    # print(i, CMC_tmp[0])
    ap_all[i] = ap_tmp
    cmc_all[i,:] = [CMC_tmp[0],CMC_tmp[4],CMC_tmp[9]]

print(list(index_all[302][:10]))

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

# np.savetxt('result/ap.txt', ap_all)
# np.savetxt('./evaluate/result/cmc_refiner.txt', cmc_all)
# np.savetxt('./evaluate/result/cmc_baseline.txt', cmc_all)
#scipy.io.savemat('result/index_base_epochlast.mat_T2wsoft13.0wsoft23.0student2.mat', {'index_base':index_all})

# # multiple-query
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# if multi:
#     for i in range(len(query_label)):
#         mquery_index1 = np.argwhere(mquery_label==query_label[i])
#         mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
#         mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
#         mq = np.mean(mquery_feature[mquery_index,:], axis=0)
#         ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#         if CMC_tmp[0]==-1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#         #print(i, CMC_tmp[0])
#     CMC = CMC.float()
#     CMC = CMC/len(query_label) #average CMC
#     print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
