########################################
### import the files
########################################
import pickle
with open('/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/embed_file/dict_embed.pickle', 'rb') as f:
    dict_embed = pickle.load(f)

from keras.models import load_model
#standard_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-standard.model")
bootstrap_model = load_model("/home/wencai/PycharmProjects/WhaleIP/mpiotte-bootstrap.model")
#model_embed = bootstrap_model.get_layer("model_1")
model_head = bootstrap_model.get_layer("head")

import pandas as pd
df_result = pd.DataFrame()
for i in range(len(dict_embed)):
    p_name_a = list(dict_embed.keys())[i]
    df_result.loc[i,"p_name"]= p_name_a
    img_embed_a = dict_embed[p_name_a]
    for j in range(len(dict_embed)):
        p_name_b = list(dict_embed.keys())[i]
        img_embed_b = dict_embed[p_name_b]
        score_ab = model_head.predict([img_embed_a,img_embed_b])[0][0]
        df_result.loc[i,p_name_b] = score_ab

print(df_result.head(20))