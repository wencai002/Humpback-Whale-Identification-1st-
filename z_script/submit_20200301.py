import pandas as pd
df_result=pd.read_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/score_result/807.csv")
df_result=df_result.drop(columns=["Unnamed: 0"])

df_submit = pd.DataFrame()
for r in range(df_result.shape[0]):
    p_name = df_result.loc[r,"p_name"]
    df_submit.loc[r, "p_name"] = p_name
    df_result_r = df_result.loc[r,:]
    df_result_r = df_result_r[1:]
    ls_r_21 = df_result_r.sort_values(ascending=False)[0:21].index
    ls_r = []
    for item in ls_r_21:
        if item != p_name:
            ls_r.append(item)
        else: pass
    for i in range(20):
        df_submit.loc[r,i] = ls_r[i]
# threshold 0.99999

df_submit.to_csv("/home/wencai/PycharmProjects/WhaleIP/Humpback-Whale-Identification-1st-/z_script/submit.csv")