#https://blog.csdn.net/hongmaodaxia/article/details/67059463
import numpy
from numpy import *
import numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import matplotlib.colors as col
import matplotlib.cm as cm

import scipy.io
import pdb

# class chj_data(object):
#     def __init__(self,data,target):
#         self.data=data
#         self.target=target

# def chj_load_file(fdata,ftarget):
#     data=numpy.loadtxt(fdata, dtype=float32)
#     target=numpy.loadtxt(ftarget, dtype=int32)

#     print(data.shape)
#     print(target.shape)
#     # pexit()

#     res=chj_data(data,target)
#     return res

# fdata="data/3.txt"
# ftarget="data/4.txt"    

# #iris = load_iris()
# iris = chj_load_file(fdata,ftarget)
def plot(filename, savefile):
       # result = scipy.io.loadmat('../../result/pytorch_result_stage2_epochlast.mat_T2wsoft10.0wsoft20.0wsoft30.0student1optimchooseadamlrmul0.1.mat')
       result = scipy.io.loadmat(filename)
       # result = scipy.io.loadmat('./evaluate/result/result_baseline.mat')
       query_feature = result['query_f']
       query_label = result['query_label'][0]
       gallery_feature = result['gallery_f']
       gallery_label = result['gallery_label'][0]

       # cmc_1 = np.loadtxt('../../result/cmc_wsoft10.0wsoft20.0wsoft30.0.txt')
       # cmc_2 = np.loadtxt('../../result/cmc_wsoft13.0wsoft20.0wsoft316.0.txt')
       cmc_1 = np.loadtxt('./evaluate/result/cmc_baseline.txt')
       cmc_2 = np.loadtxt('./evaluate/result/cmc_refiner.txt')
       query_index_increase = np.where((cmc_2 - cmc_1)[:, 0] > 0)[0]

       # cmc_3 = np.loadtxt('../../result/cmc_wsoft10.0wsoft20.0wsoft30.0_cmc20.txt')
       # query_index_cmc20 = np.where(cmc_3[:, 3] == 0)[0]
       scipy.io.savemat('./evaluate/result/query_index_increase.mat', {'query_index_increase':query_index_increase})
       # print(query_index_increase)
       num=30
       # query_index_increase = [266, 393, 548, 629, 1183, 1187, 1772, 1786, 1792, 4497, 6012,
       #                             2446, 2470, 3026, 3479, 3810, 3833, 4297]
       query_index_increase = query_index_increase[210:210+num]
       # print(query_index_increase)
       # query_index_increase = [  657, 1164,
       #        1341, 1419, 2042, 2110, 2399, 2519,
       #        2789, 3052, 4020, 4355, 4407, 4675,
       #        5415, 5496, 5704, 5733, 2863, 2544, 273, 3079,  4627,
       #        3700, 4162, 5879, 2171][5:num]#np.intersect1d(query_index_cmc20, query_index_increase)[30:]#[274,523,600,805,928,1093,1123,1125,1126,1165,1173,1151,1368,1420,2043,2111,2400]#[83,96,274,322,360,415,523,600,658,805,928,1041,1093,1123,1125,1126,1165,1173,1151,1342,1368,1420,1537,2043,2067,2111,2172,2294,2400,2413]#query_index_increase[:30]
       # query_index_increase = [ 670, 1175, 
       #        1245, 1433, 2090, 2148, 2389, 2515,
       #        2798, 3053, 4052, 4389, 4400, 4676,
       #        5420, 5487][:num]
       # query_index_increase = [ 2153, 404, 670, 1017, 
       #                      1324, 1727, 2216, 2270,
       #                      2617, 2968, 3905, 4072, 
       #                      4483, 4715, 4737, 4779][:num]

       id_show = np.unique(query_label[query_index_increase])
       query_index_show = np.where(np.isin(query_label, id_show))[0]
       gallery_index_show = np.where(np.isin(gallery_label, id_show))[0]

       query_index_show = query_index_increase

       query_feature_show = query_feature[query_index_show]
       query_label_show = query_label[query_index_show]
       gallery_feature_show = gallery_feature[gallery_index_show]
       gallery_label_show = gallery_label[gallery_index_show]
       feature_show = np.vstack((query_feature_show, gallery_feature_show))
       #print(iris.data)
       #print(iris.target)
       #exit()
       X_tsne = TSNE(n_components=2,learning_rate=10).fit_transform(feature_show)
       #X_pca = PCA().fit_transform(iris.data)
       print("finish")
       plt.figure()
       #plt.subplot(121)

       query_label_show_index = query_label_show
       gallery_label_show_index = gallery_label_show

       for i in range(len(query_label_show)):
              query_label_show_index[i] = np.where(query_label[query_index_increase] == query_label_show[i])[0][0]
       for i in range(len(gallery_label_show)):
              gallery_label_show_index[i] = np.where(query_label[query_index_increase] == gallery_label_show[i])[0][0]

       defcmap = col.LinearSegmentedColormap.from_list('mycmp',['#000000','#BEBEBE','#0000FF','#00FFFF','#00FF00','#FFFF00','#FFA500','#FF0000','#A020F0','#FF00FF','#FFB6C1','#FFE4C4','#006400','#6495ED','#9ACD32','#2F4F4F','#708090','#A52A2A','#8B0000','#8B3A62'])
       cm.register_cmap(cmap=defcmap)

       plt.scatter(X_tsne[:len(query_index_show), 0], X_tsne[:len(query_index_show), 1], cmap='mycmp', c=query_label_show_index, marker='x')
       # for i in range(len(query_index_show)):
       # 	plt.annotate("%s" % query_index_show[i], xy=(X_tsne[i, 0], X_tsne[i, 1]), fontsize=5)#, xytext=(-20, 10), textcoords='offset points')
       plt.scatter(X_tsne[len(query_index_show):, 0], X_tsne[len(query_index_show):, 1], cmap='mycmp', c=gallery_label_show_index, marker='o')#, edgecolors='r')
       #plt.subplot(122)
       #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)

       plt.xticks([])
       plt.yticks([])
       # plt.colorbar(ticks=range(20))
       plt.savefig(savefile, dpi=300)
       # plt.savefig('./evaluate/result/mhloss.png', dpi=300)
       #plt.show()


if __name__ == '__main__':
       plot(filename='./evaluate/result/result_mhloss_refiner.mat', 
              savefile='./evaluate/result/rkt.png')
       plot(filename='./evaluate/result/result_baseline.mat', 
              savefile='./evaluate/result/mhloss.png')