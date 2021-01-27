# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import parameter
import torch.optim

from .lr_scheduler import LRSchedulerWithWarmup

# def make_optimizer(cfg, model):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

#     # optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
#     optimizer = torch.optim.Adam(params, lr, betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA), eps=1e-8)
#     return optimizer


# def make_lr_scheduler(cfg, optimizer):
#     return LRSchedulerWithWarmup(
#         optimizer,
#         milestones=cfg.SOLVER.STEPS,
#         gamma=cfg.SOLVER.GAMMA,
#         warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
#         warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
#         warmup_method=cfg.SOLVER.WARMUP_METHOD,
#         mode=cfg.SOLVER.LRSCHEDULER,
#         target_lr=cfg.SOLVER.TARGET_LR,
#         power=cfg.SOLVER.POWER
#     )

def make_optimizer(model, cfg):

    '''
    Type-1
    '''
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.SOLVER.BASE_LR
    #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #     if "bias" in key:
    #         lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    #         weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # # optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    # optimizer = torch.optim.Adam(params, lr, betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA), eps=1e-8)

    '''
    Type-2
    '''
    cnn_params = list(map(id, model.visual_model.parameters()))
    lstm_params = list(map(id, model.textual_model.parameters()))
    backbone_params = cnn_params + lstm_params
    other_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    other_params = list(other_params)

    param_groups = [{'params': other_params},
    {'params': model.visual_model.parameters(), 'weight_decay': cfg.SOLVER.WEIGHT_DECAY, 'lr':cfg.SOLVER.BASE_LR},
    {'params': model.textual_model.parameters(), 'lr':cfg.SOLVER.BASE_LR}]

    optimizer = torch.optim.Adam(
        param_groups,
        lr = cfg.SOLVER.BASE_LR*10, betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA), eps=cfg.SOLVER.EPSILON
    )

    '''
    Type-3
    '''
    # add 2021年1月20日 10:45:55
    # cnn_params = list(map(id, model.visual_model.parameters()))
    # lstm_params = list(map(id, model.textual_model.parameters()))
    # backbone_params = cnn_params + lstm_params
    # other_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    # other_params = list(other_params)

    # param_groups = []
    # for key, value in model.visual_model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.SOLVER.BASE_LR
    #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #     if "bias" in key:
    #         lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    #         weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    #     param_groups += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # for key, value in model.textual_model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.SOLVER.BASE_LR
    #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #     if "bias" in key:
    #         lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    #         weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    #     param_groups += [{"params": [value], "lr": lr, "weight_decay": weight_decay}] 

    # param_groups += [{'params': other_params, "lr": cfg.SOLVER.BASE_LR*11}]
    # # param_groups = [{'params': other_params},
    # # {'params': model.visual_model.parameters(), 'weight_decay': cfg.SOLVER.WEIGHT_DECAY, 'lr': cfg.SOLVER.BASE_LR / 10},
    # # {'params': model.textual_model.parameters(), 'lr': cfg.SOLVER.BASE_LR / 10}]
    # optimizer = torch.optim.Adam(
    #     param_groups,
    #     lr = cfg.SOLVER.BASE_LR*11, betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA), eps=cfg.SOLVER.EPSILON
    # )


    '''
    Type-4
    '''
    # optimizer = torch.optim.Adadelta(model.parameters())



    return optimizer


def make_lr_scheduler(optimizer, cfg):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        mode=cfg.SOLVER.LRSCHEDULER,
        target_lr=cfg.SOLVER.TARGET_LR,
        power=cfg.SOLVER.POWER
    )