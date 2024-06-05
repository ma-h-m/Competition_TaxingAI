
import numpy as np
import torch
from torch import optim
import os,sys



from datetime import datetime

import os
import copy
import wandb
import pickle
import pandas as pd
import shutil

def copy_files_excluding(src_dir, dst_dir):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 遍历源目录中的所有文件和目录
    for root, dirs, files in os.walk(src_dir):
        # 跳过 __pycache__ 目录
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        # 如果当前目录是 run 目录，则过滤 .pkl 文件
        if os.path.basename(root) == 'run':
            files = [f for f in files if not f.endswith('.pkl')]

        # 计算相对于源目录的路径
        rel_path = os.path.relpath(root, src_dir)
        
        # 计算目标目录中的对应路径
        dst_path = os.path.join(dst_dir, rel_path)
        
        # 确保目标目录中的对应路径存在
        os.makedirs(dst_path, exist_ok=True)
        
        # 复制文件
        for file in files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_path, file)
            shutil.copy2(src_file_path, dst_file_path)
            
def _read_csv(path):
    try:
        with open(path, 'r') as f:
            df = pd.read_csv(f)
        if df.empty:
            columns = ['path', "algo", "epoch", "score"]
            df = pd.DataFrame(columns = columns)
            
    except FileNotFoundError:
            # if the file does not exist, create a new one
            columns = ['path', "algo", "epoch", "score"]
            df = pd.DataFrame(columns = columns)
    return df

# Maintaining short-term policy pool as a queue
def update_short_term_policy_pool(short_pool_size = 30,model_path = "TaxAI_modified/agents/model_self",algo = "ac", id = "Unknown", epoch = -1, government_score = 0.0, household_score = 0.0):
    log_gov_path = "TaxAI_modified/agents/model_pools/short_term/log_government.csv"
    log_household_path = "TaxAI_modified/agents/model_pools/short_term/log_household.csv"

    df_gov = _read_csv(log_gov_path)
    df_household = _read_csv(log_household_path)

    target_path = "TaxAI_modified/agents/model_pools/short_term/models/" + id + "-" + str(epoch)
    copy_files_excluding(model_path, target_path)

    new_row_gov = {"path": target_path + "/run/gov_net.pt", "algo": algo, "epoch": epoch, "score": government_score}
    new_row_household = {"path": target_path + "/run/house_net.pt", "algo": algo, "epoch": epoch, "score": household_score}

    df_gov.loc[len(df_gov)] = new_row_gov
    df_household.loc[len(df_household)] = new_row_household



    if df_gov.shape[0] > short_pool_size:
        first_gov = df_gov.iloc[0]
        df_gov = df_gov.drop(0)
        # os.remove(first_gov["path"])
        first_household = df_household.iloc[0]
        df_household = df_household.drop(0)
        # os.remove(first_household["path"])
        first_parent_dir = os.path.dirname(os.path.dirname(first_gov["path"]))
        shutil.rmtree(first_parent_dir)

    df_gov.to_csv(log_gov_path, index = False)
    df_household.to_csv(log_household_path, index = False)

# Maintaining long-term policy pool. Setting long_pool_size aims to reduce the memory usage.
def update_long_term_policy_pool(long_pool_size = 30000,model_path = "TaxAI_modified/agents/model_self",algo = "ac", id = "Unknown", epoch = -1, government_score = 0.0, household_score = 0.0):
    log_gov_path = "TaxAI_modified/agents/model_pools/long_term/log_government.csv"
    log_household_path = "TaxAI_modified/agents/model_pools/long_term/log_household.csv"

    df_gov = _read_csv(log_gov_path)
    df_household = _read_csv(log_household_path)

    
    target_path = "TaxAI_modified/agents/model_pools/long_term/models/" + id + "-" + str(epoch)
    copy_files_excluding(model_path, target_path)

    new_row_gov = {"path": target_path + "/run/gov_net.pt", "algo": algo, "epoch": epoch, "score": government_score}
    new_row_household = {"path": target_path + "/run/house_net.pt", "algo": algo, "epoch": epoch, "score": household_score}

    df_gov.loc[len(df_gov)] = new_row_gov
    df_household.loc[len(df_household)] = new_row_household



    if df_gov.shape[0] > long_pool_size:
        first_gov = df_gov.iloc[0]
        df_gov = df_gov.drop(0)
        # os.remove(first_gov["path"])
        first_household = df_household.iloc[0]
        df_household = df_household.drop(0)
        # os.remove(first_household["path"])
        first_parent_dir = os.path.dirname(os.path.dirname(first_gov["path"]))
        shutil.rmtree(first_parent_dir)

    df_gov.to_csv(log_gov_path, index = False)
    df_household.to_csv(log_household_path, index = False)

# def update_top_k_policy_pool(pool_size = 10,model_path = "TaxAI_modified/agents/model_self",algo = "ac", id = "Unknown", epoch = -1, government_score = 0.0, household_score = 0.0):
#     log_gov_path = "TaxAI_modified/agents/model_pools/top_k/log_government.csv"
#     log_household_path = "TaxAI_modified/agents/model_pools/top_k/log_household.csv"

#     df_gov = _read_csv(log_gov_path)
#     df_household = _read_csv(log_household_path)
    
#     target_path = "TaxAI_modified/agents/model_pools/top_k/models/" + id + "-" + str(epoch)
#     # copy_files_excluding(model_path, target_path)

#     new_row_gov = {"path": target_path + "/run/gov_net.pt", "algo": algo, "epoch": epoch, "score": government_score}
#     new_row_household = {"path": target_path + "/run/house_net.pt", "algo": algo, "epoch": epoch, "score": household_score}

#     df_gov.loc[len(df_gov)] = new_row_gov
#     df_household.loc[len(df_household)] = new_row_household





#     if df_gov.shape[0] > pool_size:
#         df_gov.sort_values(by = "score", ascending = False, inplace = True)
#         df_household.sort_values(by = "score", ascending = False, inplace = True)
#         last_gov = df_gov.iloc[-1]
#         last_household = df_household.iloc[-1]

#         df_gov = df_gov.drop(df_gov.tail(1).index)
#         df_household = df_household.drop(df_household.tail(1).index)

#         gov_dirs = set(df_gov["path"].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x)))))
#         household_dirs = set(df_household["path"].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x)))))
#         all_dirs = gov_dirs.union(household_dirs)


#         for dir in os.listdir("TaxAI_modified/agents/model_pools/top_k/models"):
#             if dir not in all_dirs:
#                 shutil.rmtree("TaxAI_modified/agents/model_pools/top_k/models/" + dir)

#     if (df_gov == pd.DataFrame([new_row_gov]).iloc[0]).all(axis=1).any() or (df_household == pd.DataFrame([new_row_household]).iloc[0]).all(axis=1).any():
#         copy_files_excluding(model_path, target_path)

        

#     df_gov.to_csv(log_gov_path, index = False)
#     df_household.to_csv(log_household_path, index = False)



