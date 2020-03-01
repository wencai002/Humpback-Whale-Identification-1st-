########################################
### import the files
########################################
import pickle
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/embed_file/dict_embed.pickle', 'rb') as f:
    dict_embed = pickle.load(f)
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/cropped_img_test.pickle', 'rb') as f:
    dict_cropped_img_test = pickle.load(f)

from keras.models import load_model
#standard_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-standard.model")
bootstrap_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-bootstrap.model")
#model_embed = bootstrap_model.get_layer("model_1")
model_head = bootstrap_model.get_layer("head")

import pandas as pd
df_result = pd.DataFrame()
indices = [list(range(0,100)),
           list(range(100,200)),
           list(range(200,300)),
           list(range(300,400)),
           list(range(400,500)),
           list(range(500,600)),
           list(range(600,700)),
           list(range(700,808))]

for index in indices:
    for i in index:
        p_name_a = list(dict_cropped_img_test.keys())[i]
        df_result.loc[i,"p_name"]= p_name_a
        img_embed_a = dict_embed[p_name_a]
        for j in range(len(dict_embed)):
            p_name_b = list(dict_embed.keys())[j]
            img_embed_b = dict_embed[p_name_b]
            score_ab = model_head.predict([img_embed_a,img_embed_b])[0][0]
            df_result.loc[i,p_name_b] = score_ab
        print("hey image {} is finished".format(i))
    df_result.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/score_result/{}.csv".format(i))

