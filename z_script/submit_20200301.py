import pandas as pd
df_result=pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/score_result/807.csv")
df_result=df_result.drop(columns=["Unnamed: 0"])

df_submit = pd.DataFrame()
for r in range(df_result.shape[0]):
    p_name = df_result.loc[r,"p_name"]
    df_submit.loc[r, "p_name"] = p_name
    df_result_r = df_result.loc[r,:]
    df_result_r = df_result_r[1:]
    df_result_r = df_result_r[df_result_r[1]>=0.99999]
    ls_r_21 = df_result_r.sort_values(ascending=False)[0:21]
    ls_r_21_ind = ls_r_21.index
    ls_r_8 = []
    for item in ls_r_21:
        if item != p_name:
            ls_r_8.append(item)
        else: pass
    ls_r_0 = ls_r_5_
        df_submit.loc[r,i] = ls_r_20[i]
# threshold 0.99999

df_submit.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/submit.csv")