# for n in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
# do 
#     python test.py --lambda5 ${n}
# done
config_file="./config/configs_Flickr30K.yaml"
python mytest.py --config_file ${config_file} --gpuid 2