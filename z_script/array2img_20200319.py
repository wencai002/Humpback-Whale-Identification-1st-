from keras.preprocessing.image import array_to_img
import pickle
import pandas as pd
from pathlib import Path
import os

with open("/home/wencai/PycharmProjects/WhaleIP/Keras_Martin/cropped/cropped_img_training0.pickle", "rb") as f:
    dict_cropped_img_train = pickle.load(f)
with open("/home/wencai/PycharmProjects/WhaleIP/Keras_Martin/cropped/cropped_img_training1.pickle", "rb") as f:
    dict_cropped_img_train.update(pickle.load(f))
with open("/home/wencai/PycharmProjects/WhaleIP/Keras_Martin/cropped/cropped_img_training2.pickle", "rb") as f:
    dict_cropped_img_train.update(pickle.load(f))
with open("/home/wencai/PycharmProjects/WhaleIP/Keras_Martin/cropped/cropped_img_training3.pickle", "rb") as f:
    dict_cropped_img_train.update(pickle.load(f))
with open("/home/wencai/PycharmProjects/WhaleIP/Keras_Martin/cropped/cropped_img_test.pickle", "rb") as f:
    dict_cropped_img_test = pickle.load(f)

df_train = pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/df_train.csv", index_col=0)

for j in range(len(dict_cropped_img_train)):
    p_name = list(dict_cropped_img_train.keys())[j]
    p_id = list(df_train[df_train.img==p_name]["id"])[0]
    path = Path("/home/wencai/PycharmProjects/WhaleIP/train_cropped/"+p_id)
    if not os.path.exists(path):
        os.mkdir(path)
    p_array = dict_cropped_img_train[p_name][0]
    img = array_to_img(p_array)
    img.save("/home/wencai/PycharmProjects/WhaleIP/train_cropped/{}/{}".format(p_id,p_name))

for j in range(len(dict_cropped_img_test)):
    p_name = list(dict_cropped_img_test.keys())[j]
    p_array = dict_cropped_img_test[p_name][0]
    img = array_to_img(p_array)
    img.save("/home/wencai/PycharmProjects/WhaleIP/test_cropped/{}".format(p_name))