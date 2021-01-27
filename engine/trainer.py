import matplotlib.pyplot as plt
import math
import time
import torch
import datetime
from module.utils import *
from test import test

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
    cfg 
):
    model.train()
    start_training_time = time.time()
    end_time = time.time()
    iteration = arguments['iteration']
    epoch = arguments['epoch']
    max_epoch = arguments['max_epoch']
    max_iter = len(loader_train)

    best_result = 0.0
    best_epoch = 0
    # checkpointer.save("epoch_{:d}".format(epoch), **arguments)
    while epoch < max_epoch :
        model.train()
        epoch += 1
        for step, data in enumerate(loader_train):
            
            data_time = time.time() - end_time
            iteration += 1
            arguments['iteration'] = iteration
            
            images, image_ids, txts, txt_ids, txtlen, imglabels = data
            images = images.cuda()
            image_ids = image_ids.cuda()
            txts = txts.cuda()
            txt_ids = txt_ids.cuda()
            txtlen = txtlen.cuda()
            imglabels = imglabels.cuda()

            loss_dict, prec_dict = model(images, txts, txtlen, image_ids)
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

        # Estimated Time of Arrival
        # 预计训练完成时间
        eta_seconds = meters.time.global_avg * (max_epoch * len(loader_train) - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if epoch % 1 == 0:
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
        if cfg.MODEL.SAVE:
            if epoch % cfg.MODEL.CHECKPOINT_STEP == 0:
                checkpointer.save("epoch_{:d}".format(epoch), **arguments)
            if epoch == max_epoch:
                checkpointer.save("epoch_final", **arguments)
        scheduler.step()
        
        '''
            Test
        '''
        if (epoch % cfg.MODEL.CHECKPOINT_STEP == 0) or (epoch == 1):
                logger.info('Testing')
                model.eval()
                cmc_q2g, cmc_g2q, r1, ap50 = test(model, loader_query, loader_gallery, cfg, avenorm=True)
                logger.info('cmc1_t2i = {:.4f}, cmc5_t2i = {:.4f}, cmc10_t2i = {:.4f}'.format(cmc_q2g[0], cmc_q2g[4], cmc_q2g[9]))
                logger.info('cmc1_i2t = {:.4f}, cmc5_i2t = {:.4f}, cmc10_i2t = {:.4f}'.format(cmc_g2q[0], cmc_g2q[4], cmc_g2q[9]))
                logger.info('r1 = {:.4f}, ap50 = {:.4f}'.format(r1, ap50))
                
                if cfg.MODEL.EVALUATE_METRIC == 'cmc1_t2i':
                    epoch_result = cmc_q2g[0]
                elif cfg.MODEL.EVALUATE_METRIC == 'r1':
                    epoch_result = r1
                else:
                    raise Exception('evaluate_metric wrong')
                if epoch_result > best_result:
                    best_result = epoch_result
                    best_epoch = epoch
            # if opt.save:
            #     torch.save({
            #         'epoch': epoch,
            #         'model': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            #         'best_result': best_result,
            #         'best_epoch': best_epoch,
            #         'curve_x': curve_x,
            #         'curve_y': curve_y,
            #         'time_elapsed': time_elapsed,
            #         'shadow': ema.shadow
            #     }, os.path.join(outputdir_path, 'epoch{}.pth.tar'.format(epoch)))
            #     logger.info("Save {}.pth.tar done".format(epoch))
    #     time_elapsed = time_elapsed + time.time() - since
    # logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # logger.info('best {} = {:4f} epoch {}'.format(opt.evaluate_metric, best_result, best_epoch))
    # logger.info('Save directory : [{}]'.format(outputdir_path))
    # if opt.save:
    #     copyfile(os.path.join(outputdir_path, 'epoch{}.pth.tar'.format(best_epoch)), os.path.join(outputdir_path, 'best.pth.tar'))
    #     torch.save({
    #         'epoch': epoch,
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict(),
    #         'best_result': best_result,
    #         'best_epoch': best_epoch,
    #         'curve_x': curve_x,
    #         'curve_y': curve_y,
    #         'shadow': ema.shadow,
    #         'time_elapsed': time_elapsed
    #     }, os.path.join(outputdir_path, 'epoch{}.pth.tar'.format('last')))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    logger.info(
        "Outputdir_path: {}".format(
            outputdir_path
        )
    )