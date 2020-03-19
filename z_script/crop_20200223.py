### save the cropped image in array
#######################################################
### preprocessing
### loading the cropping result file
#######################################################
import pickle

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/bounding-box_train.pickle', 'rb') as f:
    p2bb_train = pickle.load(f)
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/bounding-box_test.pickle', 'rb') as f:
    p2bb_test = pickle.load(f)

# ls = df_bb_test_raw.iloc[:,2]
# ls_1 = [item[1:-1].split(",")[0] for item in ls]
# ls_2 = [item[1:-1].split(",")[1] for item in ls]
# ls_3 = [item[1:-1].split(",")[2] for item in ls]
# ls_4 = [item[1:-1].split(",")[3] for item in ls]
# df_bb_test = pd.DataFrame()
# df_bb_test["img"]=df_bb_test_raw.iloc[:,0]
# df_bb_test["x0"]=ls_1
# df_bb_test["y0"]=ls_2
# df_bb_test["x1"]=ls_3
# df_bb_test["y1"]=ls_4

#######################################################
### preprocessing
### loading the dataframe result file
#######################################################
import pandas as pd
df_data_train = pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping_train.txt",
                            header=None)
df_data_train.rename(columns={0:"whale_id",1:"p_name",2:"size_x",3:"size_y"},inplace=True)
from PIL import Image as pil_image
# fill out the size of all pictures
for i in range(df_data_train.shape[0]):
    p_name = df_data_train.loc[i,"p_name"]
    whale_id = df_data_train.loc[i,"whale_id"]
    size_p = pil_image.open("/home/wencai/PycharmProjects/WhaleIP/train/{}/{}".format(whale_id,p_name)).size
    df_data_train.loc[i,"size_x"] = size_p[0]
    df_data_train.loc[i,"size_y"] = size_p[1]

df_data_test = pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropping_test.txt",
                            header=None)
df_data_test.rename(columns={0:"p_name",1:"size_x",2:"size_y"},inplace=True)
for i in range(df_data_test.shape[0]):
    p_name = df_data_test.loc[i,"p_name"]
    size_p = pil_image.open("/home/wencai/PycharmProjects/WhaleIP/test/{}".format(p_name)).size
    df_data_test.loc[i,"size_x"] = size_p[0]
    df_data_test.loc[i,"size_y"] = size_p[1]
########################################################
### define the different functions
########################################################
import keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
import numpy as np

img_shape = (384,384,1)
anisotropy = 2.15
crop_margin = 0.05

def read_raw_image(p_name):
    if p_name in df_data_train["p_name"].values:
        whale_id = df_data_train.loc[df_data_train["p_name"]==p_name, "whale_id"].values[0]
        img = pil_image.open("/home/wencai/PycharmProjects/WhaleIP/train/{}/{}".format(whale_id,p_name))
    else:
        img = pil_image.open("/home/wencai/PycharmProjects/WhaleIP/test/{}".format(p_name))
    return img

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0],    # rotation
                                [-np.sin(rotation),np.cos(rotation),0],
                                [0,0,1]])
    shear_matrix = np.array([[1,np.sin(shear),0],
                                [0,np.cos(shear),0],
                                [0,0,1]])
    zoom_matrix = np.array([[1.0/height_zoom,0,0],
                               [0,1.0/width_zoom,0],
                               [0,0,1]])
    shift_matrix = np.array([[1,0,-height_shift],
                             [0,1,-width_shift],
                             [0,0,1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix),
                  np.dot(zoom_matrix, shift_matrix))



def read_cropped_image(p, augment=False):
    if p in df_data_train["p_name"].values:
        size_x = df_data_train.loc[df_data_train["p_name"] == p, "size_x"].values[0]
        size_y = df_data_train.loc[df_data_train["p_name"] == p, "size_y"].values[0]
        _,(x0,y0,x1,y1) = p2bb_train[p]
    else:
        size_x = df_data_test.loc[df_data_test["p_name"] == p, "size_x"].values[0]
        size_y = df_data_test.loc[df_data_test["p_name"] == p, "size_y"].values[0]
        x0,y0,x1,y1 = p2bb_test[p]

    dx = x1-x0
    dy = y1-y0
    x0 = x0-dx*crop_margin
    x1 = x1+dx*crop_margin+1
    y0 = y0-dy*crop_margin
    y1 = y1+dy*crop_margin+1
    if (x0<0): x0=0
    if (x1>size_x): x1=size_x
    if (y0<0): y0=0
    if (y1>size_y): y1=size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy*anisotropy:
        dy = 0.5*(dx/anisotropy-dy)
        y0 = y0-dy
        y1 = y1+dy
    else:
        dx = 0.5*(dy*anisotropy-dx)
        x0 = x0-dx
        x1 = x1+dx

    # generate the transformation
    trans = np.array([[1,0,-0.5*img_shape[0]], [0,1,-0.5*img_shape[1]],[0,0,1]])
    trans = np.dot(np.array([[(y1-y0)/img_shape[0],0,0],[0,(x1-x0)/img_shape[1],0],[0,0,1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5,5),
            random.uniform(-5,5),
            random.uniform(0.8,1.0),
            random.uniform(0.8,1.0),
            random.uniform(-0.05*(y1-y0),0.05*(y1-y0)),
            random.uniform(-0.05*(x1-x0),0.05*(x1-x0))),
            trans
        )
    trans = np.dot(np.array([[1,0,0.5*(y1+y0)],[0,1,0.5*(x1+x0)],[0,0,1]]),trans)
    img = read_raw_image(p).convert("L")
    img = img_to_array(img)

    matrix = trans[:2,:2]
    offset = trans[:2,2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img,matrix,offset,output_shape=img_shape[:-1],order=1,mode="constant",cval=np.average(img))
    img = img.reshape(img_shape)
    img = img - np.mean(img,keepdims=True)
    if np.std(img,keepdims=True)==0:
        print("hey something is wrong with {}".format(p))
    img = img / np.std(img,keepdims=True) + K.epsilon()
    return img
############################################################
### define the read functions
############################################################

def read_for_training(p):
    return read_cropped_image(p,True)

def read_for_validation(p):
    return read_cropped_image(p,False)



dict_cropped_img = {}
ls_img_test = df_data_test["p_name"].tolist()
for i in range(len(ls_img_test)):
    p_name = ls_img_test[i]
    cropped_img = read_cropped_image(ls_img_test[i],False)
    cropped_img = np.expand_dims(cropped_img, axis=0)
    dict_cropped_img[p_name] = cropped_img
print("test dictionary is finished")
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_test.pickle', 'wb') as f:
    pickle.dump(dict_cropped_img,f)

ls_img_train = df_data_train["p_name"].tolist()
ls_img_train_1000 = ls_img_train[:1000]
ls_img_train_2000 = ls_img_train[1000:2000]
ls_img_train_3000 = ls_img_train[2000:3000]
ls_img_train_4000 = ls_img_train[3000:]
ls_img_train_all = [ls_img_train_1000,ls_img_train_2000,ls_img_train_3000,ls_img_train_4000]

for j in range(len(ls_img_train_all)):
    ls = ls_img_train_all[j]
    dict_cropped_img_train = {}
    for i in range(len(ls)):
        p_name = ls[i]
        cropped_img = read_cropped_image(ls[i], False)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        dict_cropped_img_train[p_name] = cropped_img
    with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_training{}.pickle'.format(j),
              'wb') as f:
        pickle.dump(dict_cropped_img_train, f)
    print("train dictionary {} is finished".format(j))

# df_result_img_a=pd.DataFrame()
# for i in range(len(ls_img_all)):
#