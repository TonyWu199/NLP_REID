# path1="/data/wuziqiang/model/CUHKPEDES/Aug/save_augmodel_augmodel=VAE_type=type1_loss=loss1_lambda5=0.1_T=0.0_LMD2=0.0_LMD3=0.0_LMD4=0.0"
# path2="/data/wuziqiang/model/CUHKPEDES/Aug/save_augmodel_augmodel=VAE_type=type1_loss=loss2_lambda5=0.1_T=0.0_LMD2=0.0_LMD3=0.0_LMD4=0.0"
# path3="/data/wuziqiang/model/CUHKPEDES/Aug/save_augmodel_augmodel=VAE_type=type2_loss=loss1_lambda5=0.1_T=0.0_LMD2=0.0_LMD3=0.0_LMD4=0.0"
# path4="/data/wuziqiang/model/CUHKPEDES/Aug/save_augmodel_augmodel=VAE_type=type2_loss=loss2_lambda5=0.1_T=0.0_LMD2=0.0_LMD3=0.0_LMD4=0.0"

# python train.py --augmodel 'VAE' --type 'type1' --loss 'loss1' --LAMBDA5 0.1 --gpuid 7 --T 8 --LAMBDA2 3 --LAMBDA3 16 --LAMBDA4 1 \
#                 --aug_model $path1 --description "aug->rkt"
# python train.py --augmodel 'VAE' --type 'type1' --loss 'loss1' --LAMBDA5 0.2 --gpuid 7 --T 8 --LAMBDA2 3 --LAMBDA3 16 --LAMBDA4 1 \
#                 --aug_model $path1 --description "aug->rkt"

# config_file="./config/configs_PRW.yaml"
config_file="./config/configs_CUHKPEDES.yaml"
# config_file="./config/configs_Flickr30K.yaml"
# config_file="./config/configs_Flowers.yaml"
# config_file="./config/configs_Birds.yaml"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --CMPC 1.0 --CMPM 1.0 --description "Birds_CMPMCMPC"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 3 --LAMBDA3 16 --LAMBDA4 0.5 --description "Birds_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 6 --LAMBDA3 16 --LAMBDA4 0.5 --description "Birds_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 9 --LAMBDA3 16 --LAMBDA4 0.5 --description "Birds_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 10 --LAMBDA3 20 --LAMBDA4 0.0 --description "Birds_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 10 --LAMBDA3 20 --LAMBDA4 0.2 --description "Birds_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 10 --LAMBDA3 20 --LAMBDA4 1.0 --description "Birds_RKT"

python train.py --config_file ${config_file} --lr 0.0002 --gpuid 7 --T 8 --LAMBDA2 3 --LAMBDA3 16 --LAMBDA4 1.0 --description "RKT_best_review"

# model_path="/data/wuziqiang/model/CUHKPEDES/fortestingNone_CUHKPEDES_resnet50_(384,128)_T=10.0_LMD1=1.0_LMD2=3.0_LMD3=16.0_LMD4=1.0_RKT/best.pth.tar"
# model_path="/data/wuziqiang/model/PRW/baseline_CN_augmodel=_type=_loss=_lambda5=0.0_T=0.0_LMD1=1.0_LMD2=0.0_LMD3=0.0_LMD4=0.0/best.pth.tar"
# model_path="/data/wuziqiang/model/PRW/RKT_testfunction_augmodel=_type=_loss=_lambda5=0.0_T=15.0_LMD1=1.0_LMD2=1.0_LMD3=11.0_LMD4=0.4/best.pth.tar"
# model_path="/data/wuziqiang/model/Flickr30K/4286_None_Flickr30K_resnet152_(224,224)_T=6.0_LMD1=1.0_LMD2=12.0_LMD3=20.0_LMD4=0.8_RKT/best.pth.tar"


# for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     python test.py --config_file ${config_file} --model_path ${model_path} --lambda5 ${n} --gpuid 7
# done
# python test.py --config_file ${config_file} --model_path ${model_path} --gpuid 7