#########################################
### import the input image data
#########################################
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
#standard_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-standard.model")
bootstrap_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-bootstrap.model")
print(bootstrap_model.summary())
#img_a = dict_cropped_img_test["PM-WWA-20170321-046.jpg"]

################################################################
### build the DataFrame
################################################################
import pandas as pd
for j in range(len(dict_cropped_img_test)):
    img_a_name = list(dict_cropped_img_test.keys())[j]
    img_a = dict_cropped_img_test[img_a_name]

    score_test_img_a = pd.DataFrame()
    for i in range(len(dict_cropped_img_test)):
        p_name = list(dict_cropped_img_test.keys())[i]
        score_test_img_a.loc[i,"p_name"] = p_name
        img_b = dict_cropped_img_test[p_name]
        score = bootstrap_model.predict([img_a,img_b])[0][0]
        score_test_img_a.loc[i,"score"] = score

    score_train0_img_a = pd.DataFrame()
    for i in range(len(dict_cropped_img_train0)):
        p_name = list(dict_cropped_img_train0.keys())[i]
        score_train0_img_a.loc[i,"p_name"] = p_name
        img_b = dict_cropped_img_train0[p_name]
        score = bootstrap_model.predict([img_a,img_b])[0][0]
        score_train0_img_a.loc[i,"score"] = score

    score_train1_img_a = pd.DataFrame()
    for i in range(len(dict_cropped_img_train1)):
        p_name = list(dict_cropped_img_train1.keys())[i]
        score_train1_img_a.loc[i,"p_name"] = p_name
        img_b = dict_cropped_img_train1[p_name]
        score = bootstrap_model.predict([img_a,img_b])[0][0]
        score_train1_img_a.loc[i,"score"] = score

    score_train2_img_a = pd.DataFrame()
    for i in range(len(dict_cropped_img_train2)):
        p_name = list(dict_cropped_img_train2.keys())[i]
        score_train2_img_a.loc[i,"p_name"] = p_name
        img_b = dict_cropped_img_train2[p_name]
        score = bootstrap_model.predict([img_a,img_b])[0][0]
        score_train2_img_a.loc[i,"score"] = score

    score_train3_img_a = pd.DataFrame()
    for i in range(len(dict_cropped_img_train3)):
        p_name = list(dict_cropped_img_train3.keys())[i]
        score_train3_img_a.loc[i,"p_name"] = p_name
        img_b = dict_cropped_img_train3[p_name]
        score = bootstrap_model.predict([img_a,img_b])[0][0]
        score_train3_img_a.loc[i,"score"] = score

    score_img_a = pd.concat([score_test_img_a,
                             score_train0_img_a,
                             score_train1_img_a,
                             score_train2_img_a,
                             score_train3_img_a])

    score_img_a.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/bootstrap_result/{}.csv".format(img_a_name))
# score_img_a = pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/img_a_score.csv")
# print(score_img_a.sort_values(by=["score"],ascending=False).head(21))
# for p_name in range(len(dict_img_all)):
#     img = dict_img_all[p_name]
#     score = standard_model.predict([img_a,img_b])[0][0]