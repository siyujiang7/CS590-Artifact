import os
import scipy.misc
from PIL import Image as image
import numpy as np
from numpy import genfromtxt
import re

def extract_number(string):
	string = string.split('.')
	return int(string[0])

imgs = os.listdir("/Users/siyujiang/Desktop/cs590/model/data/original_img/")
imgs = sorted(imgs,key=lambda x: extract_number(x))
def extract_number(string):
	string = string.split('.')
	return int(string[0])
i = 0;
for img in imgs:
 	im = scipy.misc.imread("/Users/siyujiang/Desktop/cs590/model/data/original_img/"+img, mode="RGB")
 	image_resized = scipy.misc.imresize(im, (300, 300))
 	scipy.misc.imsave("/Users/siyujiang/Desktop/cs590/model/data/train_test_data/"+img, image_resized)
