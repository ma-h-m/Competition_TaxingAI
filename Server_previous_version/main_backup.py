# This is a backup version of the server code before adding multi-threading. 
import socket
import os
from threading import Thread
import zipfile

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5001
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
import pandas as pd
# 创建服务器的根文件夹
ROOT_DIR = "Server/policy_pools"
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
import shutil
ip_to_id = {}
# def extract_info_from_path(path):
#     # 从路径中提取所需的信息
#     path_parts = path.split('/')
#     model_index = path_parts.index('models')
#     run_index = path_parts.index('run')
#     model_name = '/'.join(path_parts[model_index + 1:run_index])

#     id_part =  "_" + path_parts[model_index - 1] + "_" + model_name

#     model_relative_path = '/'.join(path_parts[7 :])
#     return model_relative_path, id_part

## especially for merge_new_csv_to_server
def extract_info_from_path2(path):
    # 从路径中提取所需的信息
    path_parts = path.split('/')
    model_index = path_parts.index('models')
    run_index = path_parts.index('run')
    model_name = '/'.join(path_parts[model_index + 1:run_index])

    id_part =  "_" + path_parts[model_index - 1] + "_" + model_name

    model_relative_path = '/'.join(path_parts[4 :])
    return model_relative_path, id_part
# def merge_csv_files():
#     government_frames = []
#     household_frames = []

#     for client_id in os.listdir(ROOT_DIR):
#         client_dir = os.path.join(ROOT_DIR, client_id)
#         if os.path.isdir(client_dir):
#             for subfolder in ['long_term', 'short_term', 'top_k']:
#                 subfolder_path = os.path.join(client_dir, subfolder)
#                 if os.path.exists(subfolder_path):
#                     gov_csv_path = os.path.join(subfolder_path, 'log_government.csv')
#                     hh_csv_path = os.path.join(subfolder_path, 'log_household.csv')
                    
#                     if os.path.exists(gov_csv_path):
#                         df_gov = pd.read_csv(gov_csv_path)
#                         df_gov['client_id'] = client_id  # 可选择添加用于识别客户端的列
#                         df_gov['path'] = df_gov['path'].apply(lambda x: os.path.join(subfolder_path, x))
#                         df_gov['path'], df_gov['id'] = zip(*df_gov['path'].apply(extract_info_from_path))
#                         df_gov['path'] = df_gov['path'].apply(lambda x: os.path.join(client_dir, x))
#                         df_gov["id"] = client_id + df_gov["id"]
#                         government_frames.append(df_gov)
#                         df_gov.insert(0, 'id', df_gov.pop('id')) 
#                     if os.path.exists(hh_csv_path):
#                         df_hh = pd.read_csv(hh_csv_path)
#                         df_hh['client_id'] = client_id  # 可选择添加用于识别客户端的列
#                         df_hh['path'] = df_hh['path'].apply(lambda x: os.path.join(subfolder_path, x))
#                         df_hh['path'], df_hh['id'] = zip(*df_hh['path'].apply(extract_info_from_path))
#                         df_hh['path'] = df_hh['path'].apply(lambda x: os.path.join(client_dir, x))
#                         df_hh["id"] = client_id + df_hh["id"]
#                         household_frames.append(df_hh)
#                         df_hh.insert(0, 'id', df_hh.pop('id'))
                        
    
#     if government_frames:
#         merged_gov_df = pd.concat(government_frames, ignore_index=True)
#         merged_gov_df.to_csv(os.path.join(ROOT_DIR, 'log_government.csv'), index=False)
    
#     if household_frames:
#         merged_hh_df = pd.concat(household_frames, ignore_index=True)
#         merged_hh_df.to_csv(os.path.join(ROOT_DIR, 'log_household.csv'), index=False)


# 定义合并新文件到现有CSV的函数

def merge_new_csv_to_server(client_id, client_dir):
    subfolders = ['long_term', 'short_term', 'top_k']
    for subfolder in subfolders:
        subfolder_path = os.path.join(client_dir, subfolder)
        if os.path.exists(subfolder_path):
            gov_csv_path = os.path.join(subfolder_path, 'log_government.csv')
            hh_csv_path = os.path.join(subfolder_path, 'log_household.csv')
            server_gov_csv_path = os.path.join(ROOT_DIR, 'log_government.csv')
            server_hh_csv_path = os.path.join(ROOT_DIR, 'log_household.csv')

            if os.path.exists(server_gov_csv_path):
                existing_gov_df = pd.read_csv(server_gov_csv_path)
            else:
                existing_gov_df = pd.DataFrame()

            if os.path.exists(server_hh_csv_path):
                existing_hh_df = pd.read_csv(server_hh_csv_path)
            else:
                existing_hh_df = pd.DataFrame()

            if os.path.exists(gov_csv_path):
                df_gov = pd.read_csv(gov_csv_path)
                df_gov['client_id'] = client_id
                df_gov['path'] = df_gov['path'].apply(lambda x: os.path.join(subfolder, x))
                df_gov['path'], df_gov['id'] = zip(*df_gov['path'].apply(extract_info_from_path2))
                df_gov['path'] = df_gov['path'].apply(lambda x: os.path.join(ROOT_DIR, client_id , x))
                df_gov["id"] = client_id + df_gov["id"]
                df_gov["evaluated_times"] = 0
                df_gov.insert(0, 'id', df_gov.pop('id')) 


                # 获取所有在CSV中记录的路径
                recorded_paths = set()

                # 处理每个路径，获取其两层父目录
                for path in df_gov['path'].tolist():
                    parent_dir = os.path.dirname(path)   # 获取第一层父目录
                    grandparent_dir = os.path.dirname(parent_dir)  # 获取第二层父目录
                    recorded_paths.add(grandparent_dir)  # 添加到集合中

                # 去除重复项
                if not existing_gov_df.empty:
                    df_gov = df_gov[~df_gov['id'].isin(existing_gov_df['id'])]

                if not existing_gov_df.empty:
                    merged_gov_df = pd.concat([existing_gov_df, df_gov], ignore_index=True)
                else:
                    merged_gov_df = df_gov

                merged_gov_df.to_csv(server_gov_csv_path, index=False)



            if os.path.exists(hh_csv_path):
                df_hh = pd.read_csv(hh_csv_path)
                df_hh['client_id'] = client_id
                df_hh['path'] = df_hh['path'].apply(lambda x: os.path.join(subfolder, x))
                df_hh['path'], df_hh['id'] = zip(*df_hh['path'].apply(extract_info_from_path2))
                df_hh['path'] = df_hh['path'].apply(lambda x: os.path.join(ROOT_DIR, client_id, x))
                df_hh["id"] = client_id + df_hh["id"]
                df_hh.insert(0, 'id', df_hh.pop('id'))
                df_hh["evaluated_times"] = 0

                # 获取所有在CSV中记录的路径


                for path in df_hh['path'].tolist():
                    parent_dir = os.path.dirname(path)  # 获取第一层父目录
                    grandparent_dir = os.path.dirname(parent_dir)  # 获取第二层父目录
                    recorded_paths.add(grandparent_dir)  # 添加到集合中

                # 去除重复项
                if not existing_hh_df.empty:
                    df_hh = df_hh[~df_hh['id'].isin(existing_hh_df['id'])]

                if not existing_hh_df.empty:
                    merged_hh_df = pd.concat([existing_hh_df, df_hh], ignore_index=True)
                else:
                    merged_hh_df = df_hh

                merged_hh_df.to_csv(server_hh_csv_path, index=False)


                models_path = os.path.join(subfolder_path, 'models')
            # 检查路径并删除未记录的文件夹
                
                for folder_name in os.listdir(models_path):
                    folder_path = os.path.join(subfolder_path,'models', folder_name)
                    # 仅处理子文件夹
                    if os.path.isdir(folder_path):
                        # 检查该子文件夹是否在recorded_paths中
                        if folder_path not in recorded_paths:
                            # 移除不在recorded_paths中的文件夹
                            shutil.rmtree(folder_path)
                            print(f'Removed folder: {folder_path}')


def handle_client(client_socket, client_address):
    print(f"[+] {client_address} connected.")
    
    client_id = ip_to_id.get(client_address[0], "Unidentified_client")
    client_dir = os.path.join(ROOT_DIR, client_id)
    
    while True:
        try:
            received = client_socket.recv(BUFFER_SIZE).decode()
            if not received:
                break
            
            command, *args = received.split(SEPARATOR)

            if command == "INIT":
                client_id = args[0]
                client_dir = os.path.join(ROOT_DIR, client_id)
                if not os.path.exists(client_dir):
                    os.makedirs(client_dir)
                print(f"Client {client_id} connected first time.")
                ip_to_id[client_address[0]] = client_id
                client_socket.send(f"ID {client_id} registered.".encode())

            elif command == "PUSH":
                if len(args) >= 2:
                    filename = args[0]
                    filesize = int(args[1])
                    filepath = os.path.join(client_dir, filename)
                    
                    # 发送确认消息
                    client_socket.send("READY".encode())
                    
                    # 确保文件夹存在
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    with open(filepath, "wb") as f:
                        bytes_received = 0
                        while bytes_received < filesize:
                            bytes_read = client_socket.recv(BUFFER_SIZE)
                            if not bytes_read:
                                break
                            f.write(bytes_read)
                            bytes_received += len(bytes_read)
                    print(f"[+] File {filename} received from {client_id} at ip address {client_address}.")

                    # 如果文件是压缩包，则解压
                    if zipfile.is_zipfile(filepath):
                        with zipfile.ZipFile(filepath, 'r') as zip_ref:
                            zip_ref.extractall(client_dir)
                        os.remove(filepath)
                    
                    # 合并新的CSV文件到服务器端的CSV文件
                    merge_new_csv_to_server(client_id, client_dir)
                    
                    client_socket.send(f"RECEIVED{SEPARATOR}{filename}".encode())
                else:
                    print("Invalid PUSH command format.")
                
            elif command == "DONE":
                if len(args) >= 1:
                    print(f"Transfer of file {args[0]} completed.")
            
            elif command == "FETCH":
                filename = args[0]
                filepath = os.path.join(client_dir, filename)
                if os.path.exists(filepath):
                    filesize = os.path.getsize(filepath)
                    client_socket.send(f"{filesize}{SEPARATOR}".encode())
                    with open(filepath, "rb") as f:
                        while True:
                            bytes_read = f.read(BUFFER_SIZE)
                            if not bytes_read:
                                break
                            client_socket.sendall(bytes_read)
                    print(f"[+] File {filename} sent to {client_address}.")
                else:
                    client_socket.send("FILE_NOT_FOUND".encode())
            
            elif command == "LIST":
                if len(args) >= 1:
                    client_id = args[0]
                    client_dir = os.path.join(ROOT_DIR, client_id)
                    files_list = []
                    for root, dirs, files in os.walk(client_dir):
                        for file in files:
                            relative_path = os.path.relpath(os.path.join(root, file), client_dir)
                            files_list.append(relative_path)
                    files_list_str = SEPARATOR.join(files_list)

                    if files_list_str == "":
                        files_list_str = "$EMPTY"

                    client_socket.send(files_list_str.encode())

            elif command == "FETCH_RANDOM_MODELS":

                # num_models = int(args[0])
                num_households = int(args[0])
                num_governments = int(args[1])
                client_id = args[2]
                
                gov_model_list, hh_model_list, gov_selected_df, hh_selected_df = get_random_models(num_households, num_governments)
                models_paths = gov_model_list + hh_model_list
                client_dir = os.path.join(ROOT_DIR, client_id)

                zip_filename = f"{client_id}_models.zip"
                zip_filepath = os.path.join(client_dir, zip_filename)
                
                def add_file_to_zip(zipf, file_path, arcname, existing_names):
                    """
                    将文件添加到 zip 文件中，避免重复文件名。
                    """
                    if arcname not in existing_names:
                        zipf.write(file_path, arcname)
                        existing_names.add(arcname)
                    else:
                        print(f"Skipping duplicate file: {arcname}")

                with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                    existing_names = set()
                    
                    # 在压缩包中创建 models 文件夹
                    models_folder = 'models/'
                    
                    # 添加模型文件到 models 文件夹中
                    for model_path in models_paths:
                        # 获取 model_path 作为一级目录
                        base_name = os.path.basename(model_path)
                        if os.path.isdir(model_path):
                            # 如果是文件夹，递归地将其内容添加到压缩文件中
                            for root, dirs, files in os.walk(model_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.join(models_folder, base_name, os.path.relpath(file_path, model_path))
                                    add_file_to_zip(zipf, file_path, arcname, existing_names)
                        else:
                            # 如果是文件，直接添加到压缩文件中，以 base_name 为一级目录
                            arcname = os.path.join(models_folder, base_name)
                            add_file_to_zip(zipf, model_path, arcname, existing_names)

                    # 将 gov_df 和 hh_df 转换为 CSV 并写入 ZIP 文件
                    gov_csv = gov_selected_df.to_csv(index=False)
                    hh_csv = hh_selected_df.to_csv(index=False)
                    
                    # 写入 gov_df
                    zipf.writestr('log_government.csv', gov_csv)
                    
                    # 写入 hh_df
                    zipf.writestr('log_household.csv', hh_csv)
                    
                    filesize = os.path.getsize(zip_filepath)
                    
                client_socket.send(f"{filesize}{SEPARATOR}".encode())
                
                # 等待客户端准备好接收文件
                client_response = client_socket.recv(BUFFER_SIZE).decode()
                if client_response == "READY":
                    with open(zip_filepath, "rb") as f:
                        while True:
                            bytes_read = f.read(BUFFER_SIZE)
                            if not bytes_read:
                                break
                            client_socket.sendall(bytes_read)
                    print(f"[+] Random models sent to {client_id} at {client_address}.")
                else:
                    print("Client not ready to receive file.")

                
                os.remove(zip_filepath)

            elif command == "FETCH_RANDOM_TOP_K_MODEL":
                
                # num_models = int(args[0])
                top_num = int(args[0])
                client_id = args[1]
                isHousehold = args[2] == "True"

                selected_path, selected_algo, selected_epoch, selected_score = get_random_top_k_model(top_num, isHousehold)

                model_path = selected_path
                client_dir = os.path.join(ROOT_DIR, client_id)

                zip_filename = f"{client_id}_models.zip"
                zip_filepath = os.path.join(client_dir, zip_filename)
                
                def add_file_to_zip(zipf, file_path, arcname, existing_names):
                    """
                    将文件添加到 zip 文件中，避免重复文件名。
                    """
                    if arcname not in existing_names:
                        zipf.write(file_path, arcname)
                        existing_names.add(arcname)
                    else:
                        print(f"Skipping duplicate file: {arcname}")

                with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                    existing_names = set()
                    
                    # 在压缩包中创建 models 文件夹
                    models_folder = 'models/'
                    
                    # 添加模型文件到 models 文件夹中
                    
                    # 获取 model_path 作为一级目录
                    base_name = os.path.basename(model_path)
                    if os.path.isdir(model_path):
                        # 如果是文件夹，递归地将其内容添加到压缩文件中
                        for root, dirs, files in os.walk(model_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, model_path)
                                add_file_to_zip(zipf, file_path, arcname, existing_names)
                    else:
                        # 如果是文件，直接添加到压缩文件中
                        arcname = os.path.basename(model_path)
                        add_file_to_zip(zipf, model_path, arcname, existing_names)



                    
                    filesize = os.path.getsize(zip_filepath)
                    
                client_socket.send(f"{filesize}{SEPARATOR}".encode())
                
                # 等待客户端准备好接收文件
                client_response = client_socket.recv(BUFFER_SIZE).decode()
                if client_response == "READY":
                    with open(zip_filepath, "rb") as f:
                        while True:
                            bytes_read = f.read(BUFFER_SIZE)
                            if not bytes_read:
                                break
                            client_socket.sendall(bytes_read)
                    print(f"[+] Random models sent to {client_id} at {client_address}.")
                else:
                    print("Client not ready to receive file.")

                
                os.remove(zip_filepath)                   

        except ConnectionResetError:
            break
    
    client_socket.close()
    print(f"[-] {client_address} disconnected.")

import random


def get_random_models(num_household, num_government):
    # 设置文件路径
    gov_log_path = os.path.join(ROOT_DIR, 'log_government.csv')
    hh_log_path = os.path.join(ROOT_DIR, 'log_household.csv')
    
    # 读取csv文件
    gov_df = pd.read_csv(gov_log_path)
    hh_df = pd.read_csv(hh_log_path)
    
    # 如果请求的数量大于数据集的数量，则调整请求数量
    if num_household > len(hh_df):
        num_household = len(hh_df)
    if num_government > len(gov_df):
        num_government = len(gov_df)
    
    # 随机抽取指定数量的模型
    gov_list = random.sample(list(gov_df['path']), num_government)
    hh_list = random.sample(list(hh_df['path']), num_household)

    # 获取抽取模型的详细信息
    gov_selected_df = gov_df[gov_df['path'].isin(gov_list)][['path', 'algo', 'epoch', 'score']]
    hh_selected_df = hh_df[hh_df['path'].isin(hh_list)][['path', 'algo', 'epoch', 'score']]
    
    gov_list = [os.path.dirname(os.path.dirname(path)) for path in gov_list]
    hh_list = [os.path.dirname(os.path.dirname(path)) for path in hh_list]

    new_prefix = 'TaxAI_modified/agents/model_pools/models_from_server'
    
    # 替换前四级目录
    def replace_prefix(path, new_prefix):
        parts = path.split(os.sep)
        if len(parts) > 4:
            return os.path.join(new_prefix, *parts[4:])
        return path
    gov_selected_df['path'] = gov_selected_df['path'].apply(replace_prefix, new_prefix=new_prefix)
    hh_selected_df['path'] = hh_selected_df['path'].apply(replace_prefix, new_prefix=new_prefix)
    
    return gov_list, hh_list, gov_selected_df, hh_selected_df

def get_random_top_k_model(top_num, isHousehold):
    # 设置文件路径
    if isHousehold:
        log_path = os.path.join(ROOT_DIR, 'log_household.csv')
    else:
        log_path = os.path.join(ROOT_DIR, 'log_government.csv')
    
    # 读取csv文件
    df = pd.read_csv(log_path)
    
    # 按照分数降序排序
    df_sorted = df.sort_values(by='score', ascending=False)
    
    # 如果top_num大于数据集的数量，则调整top_num
    if top_num > len(df_sorted):
        top_num = len(df_sorted)
    
    # 取前top_num个模型
    top_k_df = df_sorted.head(top_num)
    
    # 从前top_num个模型中随机选择一个
    selected_row = top_k_df.sample(n=1).iloc[0]
    
    selected_path = selected_row['path']
    selected_algo = selected_row['algo']
    selected_epoch = selected_row['epoch']
    selected_score = selected_row['score']
    
    return os.path.dirname(os.path.dirname(selected_path)), selected_algo, selected_epoch, selected_score

def get_random_top_k_model(top_num, isHousehold):
    # 设置文件路径
    if isHousehold:
        log_path = os.path.join(ROOT_DIR, 'log_household.csv')
    else:
        log_path = os.path.join(ROOT_DIR, 'log_government.csv')
    
    # 读取csv文件
    df = pd.read_csv(log_path)
    
    # 按照分数降序排序
    df_sorted = df.sort_values(by='score', ascending=False)
    
    # 如果top_num大于数据集的数量，则调整top_num
    if top_num > len(df_sorted):
        top_num = len(df_sorted)
    
    # 取前top_num个模型
    top_k_df = df_sorted.head(top_num)
    
    # 从前top_num个模型中随机选择一个
    selected_row = top_k_df.sample(n=1).iloc[0]
    
    selected_path = selected_row['path']
    selected_algo = selected_row['algo']
    selected_epoch = selected_row['epoch']
    selected_score = selected_row['score']
    
    return os.path.dirname(os.path.dirname(selected_path)), selected_algo, selected_epoch, selected_score

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    while True:
        client_socket, client_address = server_socket.accept()
        client_handler = Thread(target=handle_client, args=(client_socket, client_address))
        client_handler.start()

if __name__ == "__main__":
    main()
    # merge_csv_files()