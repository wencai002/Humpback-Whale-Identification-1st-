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
model_embed = bootstrap_model.get_layer("model_1")
#model_head = bootstrap_model.get_layer("head")

################################################################
### build the dictionary from embedding
################################################################

dict_embed= {}
for i in range(len(dict_cropped_img_test)):
    p_name = list(dict_cropped_img_test.keys())[i]
    img_crop = dict_cropped_img_test[p_name]
    img_embed = model_embed.predict([img_crop])
    dict_embed[p_name] = img_embed
print("dict_embed_test is finished")
for i in range(len(dict_cropped_img_train0)):
    p_name = list(dict_cropped_img_train0.keys())[i]
    img_crop = dict_cropped_img_train0[p_name]
    img_embed = model_embed.predict([img_crop])
    dict_embed[p_name] = img_embed
print("dict_embed_train0 is finished")
for i in range(len(dict_cropped_img_train1)):
    p_name = list(dict_cropped_img_train1.keys())[i]
    img_crop = dict_cropped_img_train1[p_name]
    img_embed = model_embed.predict([img_crop])
    dict_embed[p_name] = img_embed
print("dict_embed_train1 is finished")
for i in range(len(dict_cropped_img_train2)):
    p_name = list(dict_cropped_img_train2.keys())[i]
    img_crop = dict_cropped_img_train2[p_name]
    img_embed = model_embed.predict([img_crop])
    dict_embed[p_name] = img_embed
print("dict_embed_train2 is finished")
for i in range(len(dict_cropped_img_train3)):
    p_name = list(dict_cropped_img_train3.keys())[i]
    img_crop = dict_cropped_img_train3[p_name]
    img_embed = model_embed.predict([img_crop])
    dict_embed[p_name] = img_embed

with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/embed_file/dict_embed.pickle', 'wb') as f:
    pickle.dump(dict_embed,f)
print("dict_embed is finished")


#     score_img_a.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/bootstrap_result/{}.csv".format(img_a_name))
# score_img_a = pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/img_a_score.csv")
# print(score_img_a.sort_values(by=["score"],ascending=False).head(21))
# for p_name in range(len(dict_img_all)):
#     img = dict_img_all[p_name]
#     score = standard_model.predict([img_a,img_b])[0][0]