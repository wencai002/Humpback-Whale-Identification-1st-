from keras.models import load_model
standard_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-standard.model")

import pickle
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img.pickle', 'rb') as f:
    dict_cropped_img = pickle.load(f)
print(len(dict_cropped_img))

# for p_name in range(len(dict_img_all)):
#     img = dict_img_all[p_name]
#     score = standard_model.predict([img_a,img_b])[0][0]