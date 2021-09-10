import os
import argparse
import numpy as np
import torch
from config import cfg
from module.model import Network
from dataset.dataset import DatasetGallery, DatasetQuery, DatasetQuery_withtrans
from dataset.build import build_transforms
from torch.autograd import Variable
from VAE.vae import VAE, AE, AutoEncoderLayer, StackedAutoEncoder
from dataset import build_dataloader
import torch.nn.functional as F
from utils.logger import setup_logger
from tqdm import tqdm
from utils.rerank import re_ranking
# from engine.tester import test

# test the model individually
# two types: with 
# *[tranalation] 
# or
# *[VAE] 

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature_gallery(model, loader, cfg):
    features = torch.FloatTensor()
    gallery_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        img, label = data
        gallery_label = torch.cat((gallery_label, label))
        ff = []
        for i in range(2):
            # 添加图像flip操作
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            outputs = model.visual_model(input_img)    
            outputs = model.embed_model.avgpool(outputs)  
            batch_size = outputs.size(0)
            outputs = outputs.view(batch_size, -1)
            outputs = model.embed_model.visual_embed_layer(outputs)
            if cfg.MODEL.BN_LAYER:
                outputs = model.embed_model.bottelneck_global_visual(outputs)

            ff.append(outputs.data.cpu())
        ff = ff[0] + ff[1]
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features, gallery_label

def get_close_feature(outputs, train_query_feature, feature_num=5):
    '''
    Param: 
        `outputs`(Size(batch_size, dim)), features of query in test set
        `train_query_feature`(Size(train_query_size, dim)), features of query in train set
    Function:
        search the closest features with `outputs` and return
    '''
    outputs = outputs.data.cpu()

    unnorm_train_query = train_query_feature
    # 正则化
    fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
    outputs = outputs.div(fnorm.expand_as(outputs))
    fnorm = torch.norm(train_query_feature, p=2, dim=1, keepdim=True)
    train_query_feature = train_query_feature.div(fnorm.expand_as(train_query_feature))

    score = torch.mm(outputs, train_query_feature.t())
    _, index_o2t = torch.sort(score, dim=1, descending=True)

    beta= 0.9
    outputs_similar_total=torch.zeros_like(outputs)
    for i in range(feature_num):
        similar_index = index_o2t[:,i]
        outputs_similar = train_query_feature.index_select(0, similar_index)
        if i == 0:
            outputs_similar_total = outputs_similar
        else:
            outputs_similar_total = beta * outputs_similar_total + (1-beta) * outputs_similar
    # outputs_similar_total /= feature_num
    return outputs_similar_total

def getFeature(model, input_txt, input_txtlen, batch_size):
    outputs = model.textual_model(input_txt, input_txtlen)
    outputs = outputs.view(batch_size, -1)
    outputs = model.embed_model.textual_embed_layer(outputs)
    if cfg.MODEL.BN_LAYER:
        outputs = model.embed_model.bottelneck_global_textual(outputs)
    ff = outputs.data.cpu()
    return ff

def extract_feature_query(model, loader, cfg, aug_model=None, train_query_feature=None):
    features = torch.FloatTensor()
    query_label = torch.tensor([], dtype=torch.int64)
    for data in loader:
        if cfg.TEST.TYPE == '':
            txt, label, txtlen = data
            query_label = torch.cat((query_label, label))
            input_txt = Variable(txt.cuda())
            input_txtlen = Variable(txtlen.cuda())
            batch_size = input_txt.size(0)

            outputs = model.textual_model(input_txt, input_txtlen)
            outputs = outputs.view(batch_size, -1)
            outputs = model.embed_model.textual_embed_layer(outputs)
            if cfg.MODEL.BN_LAYER:
                outputs = model.embed_model.bottelneck_global_textual(outputs)
            
            outputs_norm = F.normalize(outputs, p=2, dim=1)
            
            # #添加高斯噪声
            # shape=outputs_norm.size()
            # noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
            # torch.randn(shape, out=noise)
            # outputs_norm += noise*cfg.MODEL.AUG.NOISE

            # vae augmentation module
            if aug_model != None and cfg.MODEL.AUG.LAMBDA5 != 0:
                if cfg.MODEL.AUG.AUGMODEL == 'VAE':
                    outputs_vae, mu, logvar = aug_model(outputs_norm)
                elif cfg.MODEL.AUG.AUGMODEL == 'AE' or cfg.MODEL.AUG.AUGMODEL == 'Sparse_AE':
                    encoded, outputs_vae = aug_model(outputs_norm)
                elif cfg.MODEL.AUG.AUGMODEL == 'StackAutoEncoder':
                    outputs_vae = aug_model(outputs_norm)

                outputs = (1-cfg.MODEL.AUG.LAMBDA5)*outputs + cfg.MODEL.AUG.LAMBDA5*outputs_vae

            ff = outputs.data.cpu()
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)
        elif cfg.TEST.TYPE == 'translation':
            if cfg.DATASET.NAME == 'Flickr30K':
                txt1, label, txtlen1, txt2, _, txtlen2, txt3, _, txtlen3, txt4, _, txtlen4, txt5, _, txtlen5 = data
                query_label = torch.cat((query_label, label))
                input_txt1 = Variable(txt1.cuda())
                input_txtlen1 = Variable(txtlen1.cuda())
                input_txt2 = Variable(txt2.cuda())
                input_txtlen2 = Variable(txtlen2.cuda())
                input_txt3 = Variable(txt3.cuda())
                input_txtlen3 = Variable(txtlen3.cuda())
                input_txt4 = Variable(txt4.cuda())
                input_txtlen4 = Variable(txtlen4.cuda())
                input_txt5 = Variable(txt5.cuda())
                input_txtlen5 = Variable(txtlen5.cuda())
                batch_size = input_txt1.size(0)

                
                ff1 = getFeature(model, input_txt1, input_txtlen1, batch_size)
                ff2 = getFeature(model, input_txt2, input_txtlen2, batch_size)
                ff3 = getFeature(model, input_txt3, input_txtlen3, batch_size)
                ff4 = getFeature(model, input_txt4, input_txtlen4, batch_size)
                ff5 = getFeature(model, input_txt5, input_txtlen5, batch_size)

                ff = ff1 + ff2 + ff3 + ff4 + ff5
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

                features = torch.cat((features, ff), 0)
            else:
                txt1, label, txtlen1, txt2, _, txtlen2 = data
                query_label = torch.cat((query_label, label))
                input_txt1 = Variable(txt1.cuda())
                input_txtlen1 = Variable(txtlen1.cuda())
                input_txt2 = Variable(txt2.cuda())
                input_txtlen2 = Variable(txtlen2.cuda())
                batch_size = input_txt1.size(0)

                # text1
                ff1 = getFeature(model, input_txt1, input_txtlen1, batch_size)
                # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                # ff = ff.div(fnorm.expand_as(ff))
                # features = torch.cat((features, ff), 0)

                # text2
                ff2 = getFeature(model, input_txt2, input_txtlen2, batch_size)
                # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                # ff = ff.div(fnorm.expand_as(ff))
                # features = torch.cat((features, ff), 0)

                ff = (1-cfg.MODEL.AUG.LAMBDA5)*ff1 + cfg.MODEL.AUG.LAMBDA5*ff2
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

                features = torch.cat((features, ff), 0)
        elif cfg.TEST.TYPE == 'closetrainfeature':
            txt, label, txtlen = data
            query_label = torch.cat((query_label, label))
            input_txt = Variable(txt.cuda())
            input_txtlen = Variable(txtlen.cuda())
            batch_size = input_txt.size(0)

            outputs1 = model.textual_model(input_txt, input_txtlen)
            outputs1 = outputs1.view(batch_size, -1)
            outputs1 = model.embed_model.textual_embed_layer(outputs1)
            if cfg.MODEL.BN_LAYER:
                outputs1 = model.embed_model.bottelneck_global_textual(outputs1)

            # 最近邻文本
            outputs2 = get_close_feature(outputs1, train_query_feature, 10)

            ff = (1-cfg.MODEL.AUG.LAMBDA5)*outputs1.data.cpu() + cfg.MODEL.AUG.LAMBDA5*outputs2

            # ff = ff.data.cpu()
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))    

            features = torch.cat((features, ff), 0)
    return features, query_label

def extract_train_feature(model, loader_query_train, cfg):
    features = torch.FloatTensor()
    query_label = torch.tensor([], dtype=torch.int64)
    for data in tqdm(loader_query_train):
        img, imgid, txt, txtid, txtlen = data
        query_label = torch.cat((query_label, imgid))
        input_txt = txt.cuda()
        input_txtlen = txtlen.cuda()
        batch_size = input_txt.size(0)

        outputs = model.textual_model(input_txt, input_txtlen)
        outputs = outputs.view(batch_size, -1)
        outputs = model.embed_model.textual_embed_layer(outputs)
        if cfg.MODEL.BN_LAYER:
            outputs = model.embed_model.bottelneck_global_textual(outputs)
        ff = outputs.data.cpu()
        # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        # ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features

def test(model, loader_query, loader_gallery, cfg, avenorm=True, aug_model=None, train_query_feature=None, args=None):
    if aug_model != None:
        print("Testing with aug_model")
    if cfg.TEST.TYPE == 'translation':
        print("Testing with translation")
    if cfg.TEST.TYPE == '':
        print("Normal testing")

    # 提取训练集特征
    # print('Extracting train feature')
    # train_query_feature = extract_train_feature(model, loader_query_train, cfg)
    
    query_feature, query_label = extract_feature_query(model, loader_query, cfg, aug_model, train_query_feature)
    gallery_feature, gallery_label = extract_feature_gallery(model, loader_gallery, cfg)

    # rerank
    # 计算余弦相似度，两个feature都已normalize
    # [6148, 3074]
    # q_g_score = torch.mm(query_feature, gallery_feature.t())
    # q_q_score = torch.mm(query_feature, query_feature.t())
    # g_g_score = torch.mm(gallery_feature, gallery_feature.t())
    # score = torch.from_numpy(re_ranking(q_g_score.numpy(), q_q_score.numpy(), g_g_score.numpy(), k1=args.k1, k2=args.k2, lambda_value=args.lambda_value))
    # _, index_q2g = torch.sort(score, dim=1, descending=False)
    # _, index_g2q = torch.sort(score, dim=0, descending=False)
    # no rerank
    score = torch.mm(query_feature, gallery_feature.t())
    _, index_q2g = torch.sort(score, dim=1, descending=True)
    _, index_g2q = torch.sort(score, dim=0, descending=True)

    match_q2g = (query_label.reshape(-1, 1).expand_as(score) == gallery_label[index_q2g])
    match_g2q = (gallery_label.reshape(1, -1).expand_as(score) == query_label[index_g2q])
    # print(match_g2q[:, 0].nonzero(as_tuple=False)[0][0])
    
    cmc_q2g = torch.zeros((score.shape[1]))
    for i in range(score.shape[0]):
        cmc_i = torch.zeros((score.shape[1]))
        # nonzero[0][0]返回第一个非零的索引，输出为z×n(n为维度)
        cmc_i[match_q2g[i, :].nonzero(as_tuple=False)[0][0]:] = 1
        cmc_q2g = cmc_q2g + cmc_i
    cmc_q2g = cmc_q2g / score.shape[0]

    cmc_g2q = torch.zeros((score.shape[0]))
    for i in range(score.shape[1]):
        cmc_i = torch.zeros((score.shape[0]))
        cmc_i[match_g2q[:, i].nonzero(as_tuple=False)[0][0]:] = 1
        cmc_g2q = cmc_g2q + cmc_i
    cmc_g2q = cmc_g2q / score.shape[1]
    r1, ap50 = 0, 0
    return cmc_q2g, cmc_g2q, r1, ap50

if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config/configs_CUHKPEDES.yaml', type=str, help='config parameters')
    parser.add_argument('--description', type=str, help='Modify config options using the commond-line', default=None)
    parser.add_argument('--gpuid', type=int, default=2, help='chosen gpu')
    parser.add_argument('--augmodel', type=str, default='', help='AE, VAE')
    parser.add_argument('--type', type=str, default='', help='type1, type2, type3')
    parser.add_argument('--loss', type=str, default='loss1', help='loss1, loss2')
    parser.add_argument('--lambda5', type=float, default=-1.0, help='Temperature')
    # parser.add_argument('--model_path', type=str, default='/data/wuziqiang/model/CUHKPEDES/fortestingNone_CUHKPEDES_resnet50_(384,128)_T=10.0_LMD1=1.0_LMD2=3.0_LMD3=16.0_LMD4=1.0_RKT/best.pth.tar')
    # parser.add_argument('--model_path', type=str, default="/data/wuziqiang/model/PRW/baseline_CN_augmodel=_type=_loss=_lambda5=0.0_T=0.0_LMD1=1.0_LMD2=0.0_LMD3=0.0_LMD4=0.0/best.pth.tar")
    # parser.add_argument('--model_path', type=str, default="/data/wuziqiang/model/PRW/RKT_testfunction_augmodel=_type=_loss=_lambda5=0.0_T=15.0_LMD1=1.0_LMD2=1.0_LMD3=11.0_LMD4=0.4/best.pth.tar")
    parser.add_argument('--model_path', type=str, default="/data/wuziqiang/model/Flickr30K/4286_None_Flickr30K_resnet152_(224,224)_T=6.0_LMD1=1.0_LMD2=12.0_LMD3=20.0_LMD4=0.8_RKT/best.pth.tar")
    parser.add_argument('--noise', type=float, default=0.0, help="weight of noise")
    parser.add_argument('--k1', type=int, default=10, help="param of rerank")
    parser.add_argument('--k2', type=int, default=6, help="param of rerank")
    parser.add_argument('--lambda_value', type=float, default=0.6, help="param of rerank")

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.AUG.AUGMODEL = args.augmodel
    if args.lambda5 != -1:
        cfg.MODEL.AUG.LAMBDA5 = args.lambda5
    if args.type != '':
        cfg.MODEL.AUG.TYPE = args.type
    if args.loss != '':
        cfg.MODEL.AUG.LOSS = args.loss
    if args.gpuid != 0:
        cfg.MODEL.GPUID = args.gpuid
    cfg.MODEL.AUG.NOISE = args.noise
    cfg.DATASET.TRAIN_FILE = os.path.join(cfg.DATASET.ANNO_DIR, "train.npy") 
    cfg.DATASET.TEST_FILE = os.path.join(cfg.DATASET.ANNO_DIR, "test.npy") 
    cfg.DATASET.DICTIONARY_FILE = os.path.join(cfg.DATASET.ANNO_DIR, "dictionary.npy") 
    cfg.freeze()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.GPUID)
    # Load data
    transforms_dict = build_transforms(cfg)
    dataset_gallery = DatasetGallery(cfg.DATASET.TEST_FILE, \
                                        cfg.DATASET.IMG_DIR, \
                                        transforms_dict['gallery'] \
                                    )     
    if cfg.TEST.TYPE == 'translation':
        if cfg.DATASET.NAME == 'Flickr30K':
            print('Load data for Flickr30K')
            dataset_query = DatasetQuery_withtrans(cfg.DATASET.TEST_FILE, \
                                                cfg.DATASET.DICTIONARY_FILE, \
                                                transforms_dict['query'], \
                                                datasetname=cfg.DATASET.NAME, \
                                                textNum=5,
                                                )
        else:
            dataset_query = DatasetQuery_withtrans(cfg.DATASET.TEST_FILE, \
                                                cfg.DATASET.DICTIONARY_FILE, \
                                                transforms_dict['query'], \
                                                datasetname=cfg.DATASET.NAME, \
                                                textNum=2,
                                                )     
    else:
        dataset_query = DatasetQuery(cfg.DATASET.TEST_FILE, \
                                        cfg.DATASET.DICTIONARY_FILE, \
                                        transforms_dict['query'], \
                                        datasetname=cfg.DATASET.NAME
                                        )
    loader_gallery = torch.utils.data.DataLoader(dataset_gallery, \
                                                batch_size=cfg.TEST.BATCH_SIZE, \
                                                shuffle=False, \
                                                num_workers=cfg.DATALOADER.NUM_WORKERS, \
                                                )
    loader_query = torch.utils.data.DataLoader(dataset_query, \
                                                batch_size=cfg.TEST.BATCH_SIZE, \
                                                shuffle=False, \
                                                num_workers=cfg.DATALOADER.NUM_WORKERS \
                                                )


    if cfg.MODEL.AUG.AUGMODEL != '':
        aug_model_path = os.path.join('/data/wuziqiang/model/aug', cfg.DATASET.NAME, \
                                    cfg.MODEL.AUG.AUGMODEL, \
                                    '{}_{}_{}_noise={}'.
                                    format(args.description, \
                                    cfg.MODEL.AUG.TYPE, cfg.MODEL.AUG.LOSS, cfg.MODEL.AUG.NOISE))
        if not os.path.exists(aug_model_path):
            os.mkdir(aug_model_path)
        # logger
        logger = setup_logger("---", save_dir=aug_model_path)
    else:
        logger = setup_logger("---", save_dir='./model/CUHKPEDES/translation')

    # load train query set
    dataloader = build_dataloader(cfg)
    loader_query_train = dataloader['train']

    #* Load model
    logger.info(args)
    logger.info('Loading model...')
    classnum = np.max(np.load(cfg.DATASET.TRAIN_FILE, allow_pickle=True, encoding='latin1').item()['imgid']) + 1
    # classnum = 31014
    model = Network(cfg, classnum)
    model = model.cuda()
    model_path = args.model_path
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model'], strict=False)
    model.eval()

    aug_model = None
    #* Load aug model
    if cfg.MODEL.AUG.AUGMODEL == 'AE':
        logger.info('Constructing AE model...')
        aug_model = AE()
    elif cfg.MODEL.AUG.AUGMODEL == 'Sparse_AE':
        logger.info('Constructing VAE model...')
        aug_model = AE()
    elif cfg.MODEL.AUG.AUGMODEL == 'VAE':
        logger.info('Constructing VAE model...')
        aug_model = VAE()
    elif cfg.MODEL.AUG.AUGMODEL == 'StackAutoEncoder':
        layer_list = []
        layer_list.append(AutoEncoderLayer(1024, 512, SelfTraining=True).cuda())
        layer_list.append(AutoEncoderLayer(512, 256, SelfTraining=True).cuda())
        layer_list.append(AutoEncoderLayer(256, 512, SelfTraining=True).cuda())
        layer_list.append(AutoEncoderLayer(512, 1024, SelfTraining=True).cuda())
        aug_model = StackedAutoEncoder(layer_list=layer_list)
    if aug_model:
        aug_model = aug_model.cuda()
        vae_model_dict = torch.load(aug_model_path + '/vaemodel.pth.tar')
        aug_model.load_state_dict(vae_model_dict['model'])
        # # load aug model from 
        # aug_model.load_state_dict(model_dict['aug_model'])
        # logger.info('Load aug_model Done')
        aug_model.eval()

    train_query_feature = None
    # logger.info('Extracting traing set features')
    # train_query_feature = extract_train_feature(model, loader_query_train, cfg)


    logger.info('Testing...')
    cmc_q2g, cmc_g2q, r1, ap50 = test(model, loader_query, loader_gallery, cfg, avenorm=True, aug_model=aug_model, train_query_feature=train_query_feature, args = args)
    logger.info('cmc1_t2i = {:.4f}, cmc5_t2i = {:.4f}, cmc10_t2i = {:.4f}'.format(cmc_q2g[0], cmc_q2g[4], cmc_q2g[9]))
    logger.info('cmc1_i2t = {:.4f}, cmc5_i2t = {:.4f}, cmc10_i2t = {:.4f}'.format(cmc_g2q[0], cmc_g2q[4], cmc_g2q[9]))
