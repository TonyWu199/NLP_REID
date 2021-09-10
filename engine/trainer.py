import time
from nltk.util import pr
import torch
import datetime
from module.utils import *
from .tester import test
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import os
from shutil import copyfile
from module.loss import LossComputation
from module.loss import make_loss_evaluator
from tqdm import tqdm
# import nni

# [batch_size, seq_len, embedding_dim] -> [batch_size, hidden_dim]
def txt_feature_extractor(model, input_txt, input_txtlen):
    batch_size = input_txt.size(0)
    outputs = model.textual_model(input_txt, input_txtlen)
    outputs = outputs.view(batch_size, -1)
    outputs = model.embed_model.textual_embed_layer(outputs)
    outputs = model.embed_model.bottelneck_global_textual(outputs)
    ff = outputs
    # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    # ff = ff.div(fnorm.expand_as(ff))
    return ff

# from module.embed import 
# Train and evaluate
def do_train(
    model, 
    loader_train, 
    loader_query, 
    loader_gallery, 
    optimizer, 
    scheduler, 
    outputdir_path, 
    logger, 
    meters,
    checkpointer,
    arguments,
    cfg,
    aug_model=None,
):

    model.train()
    start_training_time = time.time()
    end_time = time.time()
    iteration = arguments['iteration']
    epoch = arguments['epoch']
    max_epoch = arguments['max_epoch']
    max_iter = len(loader_train)

    epoch_result = 0.0
    best_result = 0.0
    best_epoch = 0
    best_save = {}

    # params = list(model.visual_model.parameters())
    # params += list(model.textual_model.parameters())
    # grad_clip = 2.

    # cmc_q2g, cmc_g2q, r1, ap50 = test(model, loader_query, loader_gallery, cfg, avenorm=True)
    # print('train down')
    # checkpointer.save("epoch_{:d}".format(epoch), **arguments)

    while epoch < max_epoch :
        model.train()
        epoch += 1
        loop = tqdm(enumerate(loader_train), total=len(loader_train), leave=True)

        for step, data in loop:
            
            data_time = time.time() - end_time
            iteration += 1
            arguments['iteration'] = iteration
            
            # 1 image, 1 text
            images, image_ids, txts, txt_ids, txtlen = data
            images = images.cuda()
            image_ids = image_ids.cuda()
            txts = txts.cuda()
            txt_ids = txt_ids.cuda()
            txtlen = txtlen.cuda()
            outputs_embed, loss_dict, prec_dict = model(images, txts, txtlen, image_ids)

            # images, image_ids, txts, txt_ids, txtlen, target_txts, target_ids, target_txtlen = data
            # images = images.cuda()
            # image_ids = image_ids.cuda()
            # txts = txts.cuda()
            # txt_ids = txt_ids.cuda()
            # txtlen = txtlen.cuda()
            # target_txts = target_txts.cuda()
            # target_ids = target_ids.cuda()
            # target_txtlen = target_txtlen.cuda()

            # txt2_embed = torch.tensor(0.0)
            # # with torch.no_grad():
            # txt2_embed = txt_feature_extractor(model, target_txts, target_txtlen)
            # outputs_embed, loss_dict, prec_dict = model(images, txts, txtlen, image_ids, txt2_embed, aug_model)

            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            precs = sum(prec for prec in prec_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()   

            batch_time = time.time() - end_time
            end_time = time.time()

            meters.update(loss=losses, **loss_dict)
            meters.update(prec=precs, **prec_dict)
            meters.update(time=batch_time, data=data_time)

            loop.set_description(f'Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix(loss={'ins':'{0:1.2f}'.format(loss_dict['instance_loss'].item()),\
                                    'dis':'{0:1.2f}'.format(loss_dict['distill_loss'].item()), \
                                    'mh':'{0:1.2f}'.format(loss_dict['mh_loss'].item())})

        # Estimated Time of Arrival
        # 预计训练完成时间
        eta_seconds = meters.time.global_avg * (max_epoch * len(loader_train) - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if epoch %  (cfg.MODEL.CHECKPOINT_STEP)== 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch [{epoch}/{total_epoch}]",
                        "{meters}",
                        "lr: {lr:.6f}",
                        # "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    total_epoch=max_epoch,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        # if cfg.MODEL.SAVE:
        #     if epoch % cfg.MODEL.CHECKPOINT_STEP == 0:
        #         checkpointer.save("epoch_{:d}".format(epoch), **arguments)
        #     if epoch == max_epoch:
        #         checkpointer.save("epoch_final", **arguments)
        scheduler.step()
        
        '''
            Test
        '''
        if (epoch % cfg.MODEL.CHECKPOINT_STEP == 0) or epoch == 1:
            logger.info('Testing')
            model.eval()
            cmc_q2g, cmc_g2q, r1, ap50 = test(model, loader_query, loader_gallery, cfg, avenorm=True, aug_model=aug_model)
            logger.info('cmc1_t2i = {:.4f}, cmc5_t2i = {:.4f}, cmc10_t2i = {:.4f}'.format(cmc_q2g[0], cmc_q2g[4], cmc_q2g[9]))
            logger.info('cmc1_i2t = {:.4f}, cmc5_i2t = {:.4f}, cmc10_i2t = {:.4f}'.format(cmc_g2q[0], cmc_g2q[4], cmc_g2q[9]))
            logger.info('r1 = {:.4f}, ap50 = {:.4f}'.format(r1, ap50))
            
            if cfg.MODEL.EVALUATE_METRIC == 'cmc1_t2i':
                epoch_result = cmc_q2g[0]
            else:
                raise Exception('evaluate_metric wrong')

            # #* nni module
            # nni.report_intermediate_result(epoch_result.item())

            if epoch_result > best_result:
                best_result = epoch_result
                best_epoch = epoch

                best_save = {'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_result': best_result,
                            'best_epoch': best_epoch,
                            }
                logger.info("Update best dict at epoch{}".format(epoch))
            # # epoch大于等于50才记录
            # if cfg.MODEL.SAVE and epoch >= 1:
            #     torch.save({
            #         'epoch': epoch,
            #         'model': model.state_dict(),
            #         # 'aug_model': aug_model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            #         'best_result': best_result,
            #         'best_epoch': best_epoch,
            #     }, os.path.join(outputdir_path, 'epoch{}.pth.tar'.format(epoch)))
            #     logger.info("Save {}.pth.tar done".format(epoch))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    # #* nni module
    # nni.report_final_result(best_result.item())

    logger.info('best {} = {:4f} epoch {}'.format(cfg.MODEL.EVALUATE_METRIC, best_result, best_epoch))
    logger.info('Save directory : [{}]'.format(outputdir_path))
    if cfg.MODEL.SAVE:
        torch.save(best_save, os.path.join(outputdir_path, 'best.pth.tar'))
        logger.info("Best model saved")
        # copyfile(os.path.join(outputdir_path, 'epoch{}.pth.tar'.format(best_epoch)), os.path.join(outputdir_path, 'best.pth.tar'))

    logger.info(
        "Outputdir_path: {}".format(
            outputdir_path
        )
    )