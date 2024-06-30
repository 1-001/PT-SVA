import os
import sys
path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn
import pandas as pd
import numpy as np


path = gcn.function_Le_dir()/"ml_results_single"
 
# result files
file_name_list = os.listdir(path)


result_list = [i for i in file_name_list if i[0]=="r"]
cvss_col = ["AV", "AC", "PR", "UI", "S", "C","I","A","cvss3_severity"]
estimators = ['100', '200', '300', '400', '500']  # Number of estimators for RF, XGB, LGBM
leaf_nodes = ['100', '200', '300']# Number of leaf nodes for RF, XGB, LGBM
data = pd.DataFrame()

for i in result_list:
    cur_df = pd.read_csv(path/i)
    if data.shape[0]==0:
        data = cur_df
    else:
        data = pd.concat([data,cur_df])
data = data.reset_index(drop=True)

classifier = "lgbm"
df = data[data["classifier"]==classifier]
max_mcc = 0
max_mcc_para = ""
for estimator in estimators:
    for leaf_node in leaf_nodes:
        parameters = estimator+"-"+leaf_node
        cur_df = df[df["parameters"]==parameters]
        val_mcc_list = cur_df["val_mcc"].tolist()
        val_mcc = np.mean(val_mcc_list)
        if val_mcc>max_mcc:
            max_mcc = val_mcc
            max_mcc_para = parameters
df_lgbm = df[df["parameters"]==max_mcc_para]

classifier = "rf"
df = data[data["classifier"]==classifier]
max_mcc = 0
max_mcc_para = ""
for estimator in estimators:
    for leaf_node in leaf_nodes:
        parameters = estimator+"-"+leaf_node
        cur_df = df[df["parameters"]==parameters]
        val_mcc_list = cur_df["val_mcc"].tolist()
        val_mcc = np.mean(val_mcc_list)
        if val_mcc>max_mcc:
            max_mcc = val_mcc
            max_mcc_para = parameters
df_rf = df[df["parameters"]==max_mcc_para]

df = pd.concat([df_lgbm,df_rf])
filename = gcn.function_Le_dir()/ "best_result_par.csv"
df.to_csv(filename,index = False)


