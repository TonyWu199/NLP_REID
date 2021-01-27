import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file', default='', type=str)
opt = parser.parse_args()

see = np.load(opt.file, encoding='latin1', allow_pickle=True)
print(see.item()['imglabel'])
print(see.item()['imgid'])