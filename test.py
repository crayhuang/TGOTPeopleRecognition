import glob
import os
from skimage import io, transform
import numpy as np

path = './flower_photos/'

w=100
h=100
c=3

def read_image(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []

    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
        return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
    
data, label = read_image(path)
    
print(label)