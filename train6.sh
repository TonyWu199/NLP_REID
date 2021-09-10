# # DESCRIPTION="transformer"
# # for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# # do
# #     python test.py --lambda5 ${n} --gpuid 6 --description $DESCRIPTION
# # done

# config_file="./config/configs_PRW.yaml"
config_file="./config/configs_CUHKPEDES.yaml"
# config_file="./config/configs_Flickr30K.yaml"
# config_file="./config/configs_Flowers.yaml"
python train.py --config_file ${config_file} --lr 0.0002 --gpuid 2 --T 8 --LAMBDA2 3 --LAMBDA3 8 --LAMBDA4 0.8 --description "baseline"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 6 --T 8 --LAMBDA2 3 --LAMBDA3 16 --LAMBDA4 0.5 --description "flowers_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 6 --T 8 --LAMBDA2 6 --LAMBDA3 16 --LAMBDA4 0.5 --description "flowers_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 6 --T 8 --LAMBDA2 9 --LAMBDA3 16 --LAMBDA4 0.5 --description "flowers_RKT"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 4 --T 0 --LAMBDA2 0 --LAMBDA3 0 --LAMBDA4 0.0 --description "baseline_plusmodel"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 5 --T 15 --LAMBDA2 1 --LAMBDA3 11 --LAMBDA4 0.4 --description "RKT_best_plusmodel"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 5 --T 15 --LAMBDA2 1 --LAMBDA3 11 --LAMBDA4 0.0 --description "CMKT_plusmodel"
# python train.py --config_file ${config_file} --lr 0.0002 --gpuid 6 --T 8 --LAMBDA2 3 --LAMBDA3 16 --LAMBDA4 1.0 --description "RKT_exp1"
# 
# config_file="./config/configs_PRW.yaml"
# model_path="/data/wuziqiang/model/PRW/baseline_CN_augmodel=_type=_loss=_lambda5=0.0_T=0.0_LMD1=1.0_LMD2=0.0_LMD3=0.0_LMD4=0.0/best.pth.tar"
# model_path="/data/wuziqiang/model/PRW/RKT_testfunction_augmodel=_type=_loss=_lambda5=0.0_T=15.0_LMD1=1.0_LMD2=1.0_LMD3=11.0_LMD4=0.4/best.pth.tar"

# model_path="/data/wuziqiang/model/PRW/CN_augmodel=_type=_loss=_lambda5=0.0_T=0.0_LMD1=1.0_LMD2=0.0_LMD3=0.0_LMD4=0.0/best.pth.tar"
# python train.py --config_file ${config_file} --description "CMPM_Loss"

# config_file="./config/configs_CUHKPEDES.yaml"
# for n in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     python test.py --config_file ${config_file} --model_path ${model_path} --lambda5 ${n} --gpuid 7
# done

# config_file="./config/configs_Flickr30K.yaml"

# model_path="/data/wuziqiang/model/CUHKPEDES/fortestingNone_CUHKPEDES_resnet50_(384,128)_T=10.0_LMD1=1.0_LMD2=3.0_LMD3=16.0_LMD4=1.0_RKT/best.pth.tar"
# model_path="/data/wuziqiang/model/PRW/baseline_CN_augmodel=_type=_loss=_lambda5=0.0_T=0.0_LMD1=1.0_LMD2=0.0_LMD3=0.0_LMD4=0.0/best.pth.tar"
# model_path="/data/wuziqiang/model/PRW/RKT_testfunction_augmodel=_type=_loss=_lambda5=0.0_T=15.0_LMD1=1.0_LMD2=1.0_LMD3=11.0_LMD4=0.4/best.pth.tar"
# model_path="/data/wuziqiang/model/Flickr30K/4286_None_Flickr30K_resnet152_(224,224)_T=6.0_LMD1=1.0_LMD2=12.0_LMD3=20.0_LMD4=0.8_RKT/best.pth.tar"


# for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     python test.py --config_file ${config_file} --model_path ${model_path} --lambda5 ${n} --gpuid 7
# done