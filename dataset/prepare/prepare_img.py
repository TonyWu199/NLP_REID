import os
from PIL import Image
import json
from tqdm import tqdm

inputpath = '/home/wuziqiang/data/CUHK-PEDES/CUHK_PEDES_prepare'
jsonpath = os.path.join(inputpath, 'reid_raw.json')
imgdirpath = os.path.join(inputpath, 'imgs')

with open(jsonpath, 'r') as f:
    jsondata = json.load(f)

imgpaths = []
for sample in jsondata:
    imgpaths.append([sample['file_path']])

resizeimgdirpath = os.path.join(inputpath, 'imgs_origin_python')
if not os.path.exists(resizeimgdirpath):
    print('resize image')
    for i, imgpaths_i in tqdm(enumerate(imgpaths)):
        imgpaths_i = imgpaths_i[0]
        img = Image.open(os.path.join(imgdirpath, imgpaths_i))
        # img = img.resize((256, 256))
        # img = img.resize((384, 192))
        resizeimgpath = os.path.join(resizeimgdirpath, '{}jpg'.format(imgpaths_i[:-3]))
        if not os.path.exists(os.path.dirname(resizeimgpath)):
            os.makedirs(os.path.dirname(resizeimgpath))
        img.save(resizeimgpath, 'jpeg')