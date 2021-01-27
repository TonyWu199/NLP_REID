import argparse
import os
import sys
sys.path.append('.')
from torchvision import transforms
import numpy as np

from utils.logger import setup_logger
from module.model import Network
from utils.checkpoint import Checkpointer
from dataset.dataset import *
from engine.inference import inference

def main():
    parser = argparse.ArgumentParser(description='Language Re-id Test')
    parser.add_argument('--lr', default=0.0001, type=str)
    parser.add_argument('--dataset', default='CUHK-PEDES', type=str)
    parser.add_argument('--gpuid', default='7', type=str, help='7 8')
    parser.add_argument('--maxlength', default=80, type=int)
    opt = parser.parse_args()

    # file path
    outputdir_path = os.path.join('model', '{}_maxlen={}'.format(opt.datasetname, opt.maxlength), 'Onestage_lr{}'.format(lr))
    datasetdir_path = os.path.join('dataset', '{}_maxlen={}'.format(opt.datasetname, opt.maxlength))
    dataset_test_path = os.path.join(datasetdir_path, 'test.npy')
    dictionary_path = os.path.join(datasetdir_path, 'dictionary.npy')
    imgdir_path = '/home/wuziqiang/data/CUHK-PEDES/CUHK_PEDES_prepare/imgs_256_python'

    imgmean_path = os.path.join(datasetdir_path, 'imgmean.npy')
    imgmean = np.load(imgmean_path, encoding='latin1')

    transform_list_gallery = [
        transforms.Resize(size=(224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(imgmean, [1, 1, 1])
    ]
    transform_list_query = []

    transform_query = transforms.Compose(transform_list_query)
    transform_gallery = transforms.Compose(transform_list_gallery)
    dataset_query = DatasetQuery(dataset_test_path, dictionary_path, transform_query)
    dataset_gallery = DatasetGallery(dataset_test_path, imgdir_path, transform_gallery)

    # config running environment
    num_gpus = 1
    distributed = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid

    # config logger file
    save_dir = ''
    logger = setup_logger('Language Re-id Test', save_dir=save_dir, distributed_rank=distributed)
    logger.info("Using {} GPUs".format(num_gpus))

    # define and load network
    model = Network()
    model.cuda()

    save_dir = ""
    ckpt_file = '/home/wuziqiang/code/SingleStage_LanguageReid/model/cuhkpedes_case_maxlen=80/Onestage_lr0.0001/epoch_final.pth'
    checkpointer = Checkpointer(model, save_dir=save_dir, logger=logger)
    logger.info("Loading model checkpoint...")
    _ = checkpointer.load(ckpt_file)

    inference(model, dataset_query, dataset_gallery)
