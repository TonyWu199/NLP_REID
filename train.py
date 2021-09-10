from __future__ import print_function, division
import cv2
import sys
sys.path.append('.')
import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from module.model import Network
import torch.nn as nn
from VAE.vae import VAE, AE 

from dataset.dataset import *
from dataset.sampler import build_train_sampler
from module.utils import *

from utils.metric_logger import MetricLogger, TensorboardLogger
from utils.logger import setup_logger
from engine.trainer import do_train
from solver import make_lr_scheduler, make_optimizer
from utils.checkpoint import Checkpointer

from config import cfg
from dataset import build_dataloader
# import nni
######################################################################
# Options
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='./config/configs_CUHKPEDES.yaml', type=str, help='config parameters')
parser.add_argument('--description', type=str, help='', default='')
parser.add_argument('--gpuid', type=str, default=0, help='chosen gpu')
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0)
parser.add_argument('--T', type=float, default=-1.0, help='Temperature')
parser.add_argument('--LAMBDA2', type=float, default=-1.0, help='F distillation')
parser.add_argument('--LAMBDA3', type=float, default=-1.0, help='P distillation')
parser.add_argument('--LAMBDA4', type=float, default=-1.0, help='Refine')
parser.add_argument('--CMPM', type=float, default=-1.0, help='')
parser.add_argument('--CMPC', type=float, default=-1.0, help='')


parser.add_argument('--augmodel', type=str, default='', help='AE, VAE')
parser.add_argument('--type', type=str, default='', help='type1, type2, type3')
parser.add_argument('--loss', type=str, default='', help='loss1, loss2')
parser.add_argument('--LAMBDA5', type=float, default=-1.0, help='Aug Loss')
parser.add_argument('--tester', type=str, default='single', help='')
parser.add_argument('--stage', type=str, default='augrkt', help='aug, rkt')
# parser.add_argument('--aug_model', type=str, default='/data/wuziqiang/model/CUHKPEDES/Aug/multi_augmodel=AE_type=type2_loss=loss2_lambda5=0.3/best.pth.tar')
parser.add_argument('--aug_model', type=str, default="")

args = parser.parse_args()
args = vars(args) # args包含__dict__属性，用vars可以转为dict

# nni auto search parameters, the params are stored in search_space.json
# nni_args = nni.get_next_parameter()
# args.update(nni_args)

cfg.merge_from_file(args["config_file"])

if args['T'] != -1.0:
    cfg.MODEL.LOSS.T = args['T']
if args['LAMBDA2'] != -1.0 or args['LAMBDA3'] != -1.0:
    cfg.MODEL.LOSS.LAMBDA2 = args['LAMBDA2']
    cfg.MODEL.LOSS.LAMBDA3 = args['LAMBDA3']
if args['LAMBDA4'] != -1.0:
    cfg.MODEL.LOSS.LAMBDA4 = args['LAMBDA4']
if args['gpuid'] != 0:
    cfg.MODEL.GPUID = args['gpuid']
if args['batch_size'] != 0:
    cfg.SOLVER.BATCH_SIZE = args['batch_size']
if args['augmodel'] != '':
    cfg.MODEL.AUG.AUGMODEL = args['augmodel']
if args['type'] != '':
    cfg.MODEL.AUG.TYPE = args['type']
if args['loss'] != '':
    cfg.MODEL.AUG.LOSS = args['loss']
if args['LAMBDA5'] != -1.0:
    cfg.MODEL.AUG.LAMBDA5 = args['LAMBDA5']
if args['CMPM'] != -1.0:
    cfg.MODEL.LOSS.CMPM = args['CMPM']
if args['CMPC'] != -1.0:
    cfg.MODEL.LOSS.CMPC = args['CMPC']
if args['lr'] != 0.0:
    cfg.SOLVER.BASE_LR = args['lr']
# cfg.TEST.TYPE = args.tester

cfg.DATASET.TRAIN_FILE = os.path.join(cfg.DATASET.ANNO_DIR, "train.npy") 
cfg.DATASET.TEST_FILE = os.path.join(cfg.DATASET.ANNO_DIR, "test.npy") 
cfg.DATASET.DICTIONARY_FILE = os.path.join(cfg.DATASET.ANNO_DIR, "dictionary.npy") 

if cfg.DATASET.NAME == 'CUHKPEDES':
    cfg.MODEL.VISUAL_MODEL.NAME = 'resnet50'
elif cfg.DATASET.NAME == 'PRW':
    cfg.MODEL.VISUAL_MODEL.NAME = 'resnet50'
elif cfg.DATASET.NAME == 'Flickr30K':
    cfg.MODEL.VISUAL_MODEL.NAME = 'resnet152'
cfg.freeze()
loss_dict = dict(cfg.MODEL.LOSS)
lossfunc = [loss for loss in loss_dict.keys() if loss_dict[loss] != 0.0]

# define outputdir_path according to model configuration
outputdir_path = os.path.join('/data/wuziqiang/model', '{}'.format(cfg.DATASET.NAME), \
                            '{}_T={}_LMD1={}_LMD2={}_LMD3={}_LMD4={}'\
                            .format(
                            args['description'], \
                            # cfg.MODEL.AUG.AUGMODEL,
                            # cfg.MODEL.AUG.TYPE,\
                            # cfg.MODEL.AUG.LOSS,\
                            # cfg.MODEL.AUG.LAMBDA5,\
                            cfg.MODEL.LOSS.T, \
                            cfg.MODEL.LOSS.MH, \
                            cfg.MODEL.LOSS.LAMBDA2, \
                            cfg.MODEL.LOSS.LAMBDA3, \
                            cfg.MODEL.LOSS.LAMBDA4
                            ))

if not os.path.exists(outputdir_path): 
    print('=====> Creating saveing directory <======')
    os.makedirs(outputdir_path)
# build logger
logger = setup_logger("---", save_dir=outputdir_path)
logger.info(cfg)
logger.info("Using dataset:{}".format(cfg.DATASET.ANNO_DIR))
logger.info('Output will be saved in "{}"'.format(outputdir_path))
logger.info('Loading path {}'.format(args['aug_model']))
# logger.info('\n=====> Model configuration <=====\n{}'.format(cfg))

os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.GPUID) #'6,7'

# creating dataloader
logger.info("Creating Dataloader...")
dataloader = build_dataloader(cfg)

# Train and evaluate
logger.info("Creating Network...")
classnum = np.max(np.load(cfg.DATASET.TRAIN_FILE, allow_pickle=True, encoding='latin1').item()['imgid']) + 1
logger.info('classnum : '+str(classnum))
model = Network(cfg, classnum)
model = model.cuda()

aug_model = None
if cfg.MODEL.AUG.AUGMODEL == 'VAE':
    vae_model = VAE()
    aug_model = vae_model.cuda()
elif cfg.MODEL.AUG.AUGMODEL == 'AE':
    ae_model = AE()
    aug_model = ae_model.cuda()

# load aug_model 
if args['aug_model'] != "":
    logger.info('loading model')
    model_path = args['aug_model'] + '/best.pth.tar'
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model'])
    aug_model.load_state_dict(model_dict['aug_model'])
# if args['stage'] == 'augrkt':
#     logger.info('augmodel fixed')
#     aug_model.eval()


if aug_model != None:
    logger.info("Total parameters； {:.2f}M".\
                format((sum(p.numel() for p in model.parameters()) + \
                    sum(p.numel() for p in aug_model.parameters())) / 1000000.0))
else:
    logger.info("Total parameters； {:.2f}M".\
                format(sum(p.numel() for p in model.parameters()) / 1000000.0))

logger.info("Creating Solver...")
optimizer = make_optimizer(model, cfg, aug_model)
scheduler = make_lr_scheduler(optimizer, cfg)

# checkpoint setting
# output_dir = os.path.join(root_path, outputdir_path)
output_dir = outputdir_path
checkpointer = Checkpointer(
    model, optimizer, scheduler, outputdir_path, save_to_disk=True
)

meters = MetricLogger(delimiter="  ")
# pack some parameters 
# to save in ckpt 
arguments = {}
arguments['iteration'] = 0
arguments['epoch'] = 0
arguments['max_epoch'] = cfg.SOLVER.NUM_EPOCHES

do_train(
    model, 
    dataloader['train'], 
    dataloader['query'], 
    dataloader['gallery'], 
    optimizer,
    scheduler, 
    outputdir_path, 
    logger, 
    meters, 
    checkpointer, 
    arguments,
    cfg,
    aug_model
)

