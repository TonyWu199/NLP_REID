import os

from yacs.config import CfgNode as CN

_C = CN()

# *--------------------------------
# *DATASET
# *--------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = "CUHKPEDES"
_C.DATASET.IMG_DIR = "/home/wuziqiang/data/CUHK-PEDES/CUHK_PEDES_prepare/imgs_256_python"
_C.DATASET.ANNO_DIR = "./dataset/cuhkpedes/"
_C.DATASET.TRAIN_FILE = os.path.join(_C.DATASET.ANNO_DIR, "train.npy") 
_C.DATASET.TEST_FILE = os.path.join(_C.DATASET.ANNO_DIR, "test.npy") 
_C.DATASET.DICTIONARY_FILE = os.path.join(_C.DATASET.ANNO_DIR, "dictionary.npy") 
# _C.DATASET.MEAN_FILE = os.path.join(_C.DATASET.ANNO_DIR, "imgmean.npy") 

# *--------------------------------
# *DATALOADER
# *--------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8

# *--------------------------------
# *INPUT
# *--------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 224
_C.INPUT.WIDTH = 224
_C.INPUT.PIXEL_MEAN =[0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# *--------------------------------
# *MODEL
# *--------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = 'CUDA'
_C.MODEL.GPUID = 6
_C.MODEL.EVALUATE_METRIC = 'cmc1_t2i'  #cmc1_t2i, r1
_C.MODEL.CHECKPOINT_STEP = 10
_C.MODEL.SAVE = False
_C.MODEL.EMBEDDING_SIZE = 1024
_C.MODEL.BN_LAYER = False

# *--------------------------------
# *LOSS
# *--------------------------------
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.INSTANCE = 0.0
_C.MODEL.LOSS.GLOBALALIGN = 0.0
_C.MODEL.LOSS.CMPM = 0.0
_C.MODEL.LOSS.CMPC = 0.0
_C.MODEL.LOSS.MH = 0.0

# *--------------------------------
# *IMAGE_MODEL
# *--------------------------------
_C.MODEL.VISUAL_MODEL = CN()
_C.MODEL.VISUAL_MODEL.NAME = 'resnet50'
_C.MODEL.VISUAL_MODEL.RES5_STRIDE = 2
_C.MODEL.VISUAL_MODEL.RES5_DILATION = 1
_C.MODEL.VISUAL_MODEL.USE_C4 = False
_C.MODEL.VISUAL_MODEL.NUM_STRIPES = 6

# *--------------------------------
# *LANGUAGE_MODEL
# *--------------------------------
_C.MODEL.TEXTUAL_MODEL = CN()
_C.MODEL.TEXTUAL_MODEL.NAME = 'bilstm'
_C.MODEL.TEXTUAL_MODEL.MAX_LENGTH = 80
_C.MODEL.TEXTUAL_MODEL.EMBEDDING_SIZE = 768
_C.MODEL.TEXTUAL_MODEL.HIDDEN_SIZE = 512
_C.MODEL.TEXTUAL_MODEL.DROPOUT = 0.0
_C.MODEL.TEXTUAL_MODEL.BIDIRECTION = True
_C.MODEL.TEXTUAL_MODEL.NUM_LAYERS = 1
_C.MODEL.TEXTUAL_MODEL.ATN_LAYERS = ''
_C.MODEL.TEXTUAL_MODEL.WORDS = False
# *--------------------------------
# *SOLVER
# *--------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_EPOCHES = 80

_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.EPSILON = 1e-8

_C.SOLVER.LRSCHEDULER = "step"

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500, )

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001

# *--------------------------------
# *TEST
# *--------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
