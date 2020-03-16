### save the boundbox coordinate
################################################################
# import sys
# old_stderr = sys.stderr
# sys.stderr = open('/dev/null', 'w')
#
# sys.stderr = old_stderr
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from PIL import Image as pil_image
from PIL.ImageDraw import Draw
from os.path import isfile

def read_raw_image(p):
    return pil_image.open(p)

###############################
### Relevant functions
###############################
img_shape  = (128,128,1)
anisotropy = 2.15

import random
import numpy as np
from scipy.ndimage import affine_transform
from keras.preprocessing.image import img_to_array

def read_array(p):
    img = read_raw_image(p).convert('L')
    return img_to_array(img)

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def center_transform(affine, input_shape):
    hi, wi = float(input_shape[0]), float(input_shape[1])
    ho, wo = float(img_shape[0]), float(img_shape[1])
    top, left, bottom, right = 0, 0, hi, wi
    if wi/hi/anisotropy < wo/ho: # input image too narrow, extend width
        w     = hi*wo/ho*anisotropy
        left  = (wi-w)/2
        right = left + w
    else: # input image too wide, extend height
        h      = wi*ho/wo/anisotropy
        top    = (hi-h)/2
        bottom = top + h
    center_matrix   = np.array([[1, 0, -ho/2], [0, 1, -wo/2], [0, 0, 1]])
    scale_matrix    = np.array([[(bottom - top)/ho, 0, 0], [0, (right - left)/wo, 0], [0, 0, 1]])
    decenter_matrix = np.array([[1, 0, hi/2], [0, 1, wi/2], [0, 0, 1]])
    return np.dot(np.dot(decenter_matrix, scale_matrix), np.dot(affine, center_matrix))

def transform_img(x, affine):
    matrix   = affine[:2,:2]
    offset   = affine[:2,2]
    x        = np.moveaxis(x, -1, 0)
    channels = [affine_transform(channel, matrix, offset, output_shape=img_shape[:-1], order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    return np.moveaxis(np.stack(channels, axis=0), 0, -1)

def read_for_validation(p):
    x  = read_array(p)
    t  = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t

def read_for_training(p):
    x  = read_array(p)
    t  = build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.9, 1.0),
            random.uniform(0.9, 1.0),
            random.uniform(-0.05*img_shape[0], 0.05*img_shape[0]),
            random.uniform(-0.05*img_shape[1], 0.05*img_shape[1]))
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t

def coord_transform(list, trans):
    result = []
    for x,y in list:
        y,x,_ = trans.dot([y,x,1]).astype(np.int)
        result.append((x,y))
    return result

def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x,y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0,y0,x1,y1

import matplotlib.pyplot as plt
def show_whale(imgs, per_row=5):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))
    for ax in axes.flatten(): ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))

##########################################################################################
### Now is the real processing
### first load the model and data
##########################################################################################
from numpy import loadtxt
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from tqdm import tqdm, tqdm_notebook
from numpy.linalg import inv as mat_inv

crop_model = load_model("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping.model")

from pathlib import Path
train_data_folder = Path ("/home/wencai/PycharmProjects/WhaleIP/train")
test_data_folder = Path("/home/wencai/PycharmProjects/WhaleIP/test")

import pandas as pd
df_p_train_names = pd.DataFrame(columns=["id","p_names"])
folders_train = list(train_data_folder.glob("*"))
for folder_train in folders_train:
    folder_train_name = str(folder_train).split("/")[6]
    # if folder_train_name != "-1":
    files = list(folder_train.glob("*"))
    for file in files:
        p_name = str(file).split("/")[7]
        df_p_train_names = pd.concat([df_p_train_names,
                                      pd.DataFrame({"id":[folder_train_name],"p_names":[p_name]})])
df_p_train_names["x"]=int(0)
df_p_train_names["y"]=int(0)
df_p_train_names.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping_train.txt",
                  header=None, index=None, sep=",")

p_test_names = []
files_test = list(test_data_folder.glob("*"))
for file in files_test:
    p_name = str(file).split("/")[6]
    p_test_names.append(p_name)

import numpy as np
df_p_test_names = pd.DataFrame()
df_p_test_names["p_names"]=p_test_names
df_p_test_names["y"]=np.zeros(len(p_test_names),dtype=int)
df_p_test_names["x"]=np.zeros(len(p_test_names),dtype=int)
df_p_test_names.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping_test.txt",
                  header=None, index=None, sep=",")

from keras import backend as K
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping_train.txt', 'rt') as f:
    data_train = f.read().split('\n')[:-1]
data_train = [line.split(',') for line in data_train]
data_train = [(id,p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for id,p,*coord in data_train]
data_a_train = np.zeros((len(data_train),)+img_shape,dtype=K.floatx())
for i,(id,p,_) in enumerate(data_train):
    img,trans       = read_for_validation("/home/wencai/PycharmProjects/WhaleIP/train/{}/{}".format(id,p))
    data_a_train[i,:,:,:] = img

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping_test.txt', 'rt') as f:
    data_test = f.read().split('\n')[:-1]
data_test = [line.split(',') for line in data_test]
data_test = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data_test]
data_a_test = np.zeros((len(data_test),)+img_shape,dtype=K.floatx())
for i,(p,_) in enumerate(data_test):
    img,trans       = read_for_validation("/home/wencai/PycharmProjects/WhaleIP/test/"+p)
    data_a_test[i,:,:,:] = img



#############################################
### the real transform
#############################################
p2bb_train = {}
for id_p,p,_ in data_train:
    if p not in p2bb_train:
        img,trans = read_for_validation("/home/wencai/PycharmProjects/WhaleIP/train/{}/{}".format(id_p,p))
        a         = np.expand_dims(img, axis=0)
        x0, y0, x1, y1     = crop_model.predict(a).squeeze()
        (u0, v0), (u1, v1) = coord_transform([(x0,y0),(x1,y1)], trans)
        p2bb_train[p]      = (id_p,(u0, v0, u1, v1))

p2bb_test = {}
for p in p_test_names:
    if p not in p2bb_test:
        img,trans         = read_for_validation("/home/wencai/PycharmProjects/WhaleIP/test/{}".format(p))
        a                 = np.expand_dims(img, axis=0)
        x0, y0, x1, y1    = crop_model.predict(a).squeeze()
        (u0, v0),(u1, v1) = coord_transform([(x0,y0),(x1,y1)], trans)
        p2bb_test[p]      = (u0, v0, u1, v1)

import pickle
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/bounding-box_train.pickle', 'wb') as f:
    pickle.dump(p2bb_train, f)

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/bounding-box_test.pickle', 'wb') as f:
    pickle.dump(p2bb_test, f)

###################################################
### show some examples
###################################################

# images = []
# for i,(id,p,_) in enumerate(data_train[:25]):
#     a         = data_a_train[i:i+1]
#     rect      = crop_model.predict(a).squeeze()
#     img       = array_to_img(a[0]).convert('RGB')
#     draw      = Draw(img)
#     draw.rectangle(rect, outline='yellow')
#     images.append(img)
# images[10].show()
# images[9].show()
# images[11].show()
# images[12].show()


