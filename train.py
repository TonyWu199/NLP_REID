from __future__ import print_function, division
import sys
sys.path.append('.')
import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from module.model import Network
# from shutil import copyfile

from dataset.dataset import *
from dataset.sampler import *
from module.utils import *
# import torchsnooper

from utils.metric_logger import MetricLogger, TensorboardLogger
from utils.logger import setup_logger
from engine.trainer import do_train
from solver import make_lr_scheduler, make_optimizer
from utils.checkpoint import Checkpointer

from config import cfg
from dataset import build_dataloader
######################################################################
# Options
parser = argparse.ArgumentParser()
# parser.add_argument('--method',default='cls', type=str)
# parser.add_argument('--margin',default=0.3, type=int)
# parser.add_argument('--w_1_1', default=1, type=float) # ins_loss_image
# parser.add_argument('--w_1_2', default=1, type=float) # ins_loss_text
# parser.add_argument('--w_2',   default=0.2, type=float) # mh_loss
# parser.add_argument('--w_3_1', default=3, type=float) # distill_loss_F
# parser.add_argument('--w_3_2', default=16, type=float) # distill_loss_p
# parser.add_argument('--w_4',   default=0.6, type=float) # label_loss
# parser.add_argument('--w_5',   default=0, type=float) # contrast_loss
# parser.add_argument('--w_6',   default=0, type=float) # hardtriplet_loss
# parser.add_argument('--student', default='txt', type=str)
# parser.add_argument('--temperature', default=8, type=float)
# parser.add_argument('--num_instances', default=4, type=int)
# parser.add_argument('--resume', type=int, default=0)

parser.add_argument('--config_file', default='./config/configs.yaml', type=str, help='config parameters')
args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.freeze()

loss_dict = dict(cfg.MODEL.LOSS)
lossfunc = [loss for loss in loss_dict.keys() if loss_dict[loss] != 0.0]

# define outputdir_path according to model configuration
outputdir_path = os.path.join('model', '{}'\
                              .format(cfg.DATASET.NAME), \
                              't2i_i2t_Size({},{})_Vmodel={}_Tmodel={}_Loss=({})_GRA=({})' \
                              .format(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, \
                              cfg.MODEL.VISUAL_MODEL.NAME, \
                              cfg.MODEL.TEXTUAL_MODEL.NAME, \
                              '+'.join(lossfunc), \
                              cfg.MODEL.VISUAL_MODEL.NUM_STRIPES
                              ))

if not os.path.exists(outputdir_path): 
    print('=====> Creating saveing directory <======')
    os.makedirs(outputdir_path)
# build logger
logger = setup_logger("---", save_dir=outputdir_path)
logger.info('Output will be saved in "{}"'.format(outputdir_path))
# logger.info('\n=====> Model configuration <=====\n{}'.format(cfg))

os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.GPUID)

# creating dataloader
logger.info("Creating Dataloader...")
dataloader = build_dataloader(cfg)

# Train and evaluate
logger.info("Creating Network...")
classnum = np.max(np.load(cfg.DATASET.TRAIN_FILE, allow_pickle=True, encoding='latin1').item()['imgid']) + 1
model = Network(cfg)
model = model.cuda()

logger.info("Total parameters； {:.2f}M".\
            format(sum(p.numel() for p in model.parameters()) / 1000000.0))

logger.info("Creating Solver...")
optimizer = make_optimizer(model, cfg)
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
    cfg
)

