import torch
from torchvision import transforms
from torch.utils.data import Dataset
from .dataset import DatasetTrain, DatasetQuery, DatasetGallery
import os
print(os.getcwd())
def build_transforms(cfg):
    # parameters:
    #    cfg: config parameters
    # return:
    #    transform of train, query, gallery

    transform_list_train = transforms.Compose([
        # transforms.RandomResizedCrop(size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH), 
        #                             scale=(0.8,1.0), 
        #                             ratio=(0.75,1.3333), 
        #                             interpolation=3),#Image.BICUBIC)
        transforms.Resize(size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    ])

    transform_list_gallery = transforms.Compose([
        transforms.Resize(size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    ])
    transform_list_query = transforms.Compose([])

    transform_dict = {}
    transform_dict.update({'train':transform_list_train})
    transform_dict.update({'query':transform_list_query})
    transform_dict.update({'gallery':transform_list_gallery})

    return transform_dict


def build_dataloader(cfg):
    # parameters:
    #    cfg: config parameters
    # return:
    #    loader dict
    transforms_dict = build_transforms(cfg)

    dataset_train = DatasetTrain(cfg.DATASET.TRAIN_FILE, \
                                 cfg.DATASET.DICTIONARY_FILE, \
                                 cfg.DATASET.IMG_DIR, \
                                 transforms_dict['train'] \
                                )
    dataset_gallery = DatasetGallery(cfg.DATASET.TEST_FILE, \
                                     cfg.DATASET.IMG_DIR, \
                                     transforms_dict['gallery'] \
                                    )

    dataset_query = DatasetQuery(cfg.DATASET.TEST_FILE, \
                                 cfg.DATASET.DICTIONARY_FILE, \
                                 transforms_dict['query'] \
                                )         

    loader_train = torch.utils.data.DataLoader(dataset_train, \
                                               batch_size=cfg.SOLVER.BATCH_SIZE, \
                                               shuffle=True, \
                                               num_workers=cfg.DATALOADER.NUM_WORKERS, \
                                               drop_last=True \
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
    dataloader = {}
    dataloader.update({'train': loader_train})
    dataloader.update({'gallery': loader_gallery})
    dataloader.update({'query': loader_query})

    return dataloader
