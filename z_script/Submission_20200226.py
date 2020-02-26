import pickle
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_test.pickle', 'rb') as f:
    dict_cropped_img_test = pickle.load(f)
print(len(dict_cropped_img_test))

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_training0.pickle', 'rb') as f:
    dict_cropped_img_train0 = pickle.load(f)
print(len(dict_cropped_img_train0))

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_training1.pickle', 'rb') as f:
    dict_cropped_img_train1 = pickle.load(f)
print(len(dict_cropped_img_train1))

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_training2.pickle', 'rb') as f:
    dict_cropped_img_train2 = pickle.load(f)
print(len(dict_cropped_img_train2))

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_training3.pickle', 'rb') as f:
    dict_cropped_img_train3 = pickle.load(f)
print(len(dict_cropped_img_train3))

from keras.models import load_model
standard_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-standard.model")
img_a = dict_cropped_img_test["PM-WWA-20170321-046.jpg"]

import pandas as pd
score_img_a = pd.DataFrame()
for i in range(len(dict_cropped_img_test)):
    p_name = list(dict_cropped_img_test.keys())[i]
    score_img_a.loc[i,"p_name"] = p_name
    img_b = dict_cropped_img_test[p_name]
    score = standard_model.predict([img_a,img_b])[0][0]
    score_img_a.loc[i,"score"] = score

score_img_a.to_csv("/home/wencai/PycharmProjects/WhaleIP/img_a_score.csv")
print(score_img_a.head(30))
# for p_name in range(len(dict_img_all)):
#     img = dict_img_all[p_name]
#     score = standard_model.predict([img_a,img_b])[0][0]