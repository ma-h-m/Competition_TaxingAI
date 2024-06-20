import socket
import os
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor
import zipfile

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5002
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
import pandas as pd
# 创建服务器的根文件夹
ROOT_DIR = "Server/policy_pools"
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
import shutil
ip_to_id = {}
file_lock = Lock()

user_info_path = os.path.join(ROOT_DIR, 'user_info.csv')



def handle_client(client_socket, client_address):
    print(f"[+] {client_address} connected.")
    
    client_id = ip_to_id.get(client_address[0], "Unidentified_client")
    receive_buffer_path = os.path.join(ROOT_DIR, "receive_buffer")
    if not os.path.exists(receive_buffer_path):
        os.makedirs(receive_buffer_path)
    
    while True:
        try:
            received = client_socket.recv(BUFFER_SIZE).decode()
            if not received:
                break
            
            command, *args = received.split(SEPARATOR)

            if command == "INIT":
                client_id = args[0]
                
                
                with file_lock:
                    if not os.path.exists(user_info_path):
                        user_info = pd.DataFrame(columns=['id'])
                        user_info.to_csv(user_info_path, index=False)

                    user_info = pd.read_csv(user_info_path)
                    if client_id not in user_info['id'].values:
                        user_info = user_info.append({'id': client_id}, ignore_index=True)
                        user_info.to_csv(user_info_path, index=False)
                            
                print(f"Client {client_id} connected first time.")
                ip_to_id[client_address[0]] = client_id
                client_socket.send(f"ID {client_id} registered.".encode())

            elif command == "PUSH":
                if len(args) >= 2:
                    filename = args[0]
                    filesize = int(args[1])
                    model_id = args[2]
                    algo_name = args[3]
                    epoch = int(args[4])
                    filepath = os.path.join(ROOT_DIR, client_id + "_" + model_id + ".zip")
                    
                    # 发送确认消息
                    client_socket.send("READY".encode())
                    
                    # 确保文件夹存在
                    with file_lock:
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
                        model_path = ""
                        # 如果文件是压缩包，则解压
                        if zipfile.is_zipfile(filepath):
                            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                                top_level_name = zip_ref.namelist()[0].split('/')[0]
                                model_path = os.path.join(ROOT_DIR, top_level_name)
                                zip_ref.extractall(ROOT_DIR)
                            os.remove(filepath)
                    
                    # 更新服务器端log文件
                    gov_log_path = os.path.join(ROOT_DIR, 'log_government.csv')
                    hh_log_path = os.path.join(ROOT_DIR, 'log_household.csv')
                    if not os.path.exists(gov_log_path):
                        gov_df = pd.DataFrame(columns=['id','path', 'algo', 'epoch', 'score', 'client_id', 'evaluated_times'])
                        gov_df.to_csv(gov_log_path, index=False)
                    if not os.path.exists(hh_log_path):
                        hh_df = pd.DataFrame(columns=['id','path', 'algo', 'epoch', 'score', 'client_id', 'evaluated_times'])
                        hh_df.to_csv(hh_log_path, index=False)
                    
                    gov_df = pd.read_csv(gov_log_path)
                    hh_df = pd.read_csv(hh_log_path)
                    # gov_df.append({'id': model_id, 'path': filepath, 'algo': algo_name, 'epoch': epoch, 'score': 0, 'client_id': client_id, 'evaluated_times': 0}, ignore_index=True)
                    new_row_gov = {'id': model_id, 'path': os.path.join(model_path,"run/gov_net.pt"), 'algo': algo_name, 'epoch': epoch, 'score': 0, 'client_id': client_id, 'evaluated_times': 0}
                    gov_df.loc[len(gov_df)] = new_row_gov

                    gov_df.to_csv(gov_log_path, index=False)
                    # hh_df.append({'id': model_id, 'path': filepath, 'algo': algo_name, 'epoch': epoch, 'score': 0, 'client_id': client_id, 'evaluated_times': 0}, ignore_index=True)

                    new_row_hh = {'id': model_id, 'path': os.path.join(model_path, "run/house_net.pt"), 'algo': algo_name, 'epoch': epoch, 'score': 0, 'client_id': client_id, 'evaluated_times': 0}
                    hh_df.loc[len(hh_df)] = new_row_hh
                    hh_df.to_csv(hh_log_path, index=False)

                    client_socket.send(f"RECEIVED{SEPARATOR}{filename}".encode())
                else:
                    print("Invalid PUSH command format.")
                
            elif command == "DONE":
                if len(args) >= 1:
                    print(f"Transfer of file {args[0]} completed.")
            
            # elif command == "FETCH":
            #     filename = args[0]
            #     filepath = os.path.join(client_dir, filename)
            #     if os.path.exists(filepath):
            #         filesize = os.path.getsize(filepath)
            #         client_socket.send(f"{filesize}{SEPARATOR}".encode())
            #         with open(filepath, "rb") as f:
            #             while True:
            #                 bytes_read = f.read(BUFFER_SIZE)
            #                 if not bytes_read:
            #                     break
            #                 client_socket.sendall(bytes_read)
            #         print(f"[+] File {filename} sent to {client_address}.")
            #     else:
            #         client_socket.send("FILE_NOT_FOUND".encode())
            
            elif command == "LIST":
                if len(args) >= 1:
                    client_id = args[0]
                    client_dir = os.path.join(ROOT_DIR, client_id)
                    files_list = []
                    with file_lock:
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
                with file_lock:
                    gov_model_list, hh_model_list, gov_selected_df, hh_selected_df = get_random_models(num_households, num_governments)
                    models_paths = gov_model_list + hh_model_list
                    # client_dir = os.path.join(ROOT_DIR, client_id)

                    zip_filename = f"{client_id}_models.zip"
                    zip_filepath = os.path.join(receive_buffer_path, zip_filename)
                    
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

                with file_lock:
                    os.remove(zip_filepath)

            elif command == "FETCH_RANDOM_TOP_K_MODEL":
                
                
                top_num = int(args[0])
                client_id = args[1]
                isHousehold = args[2] == "True"

                selected_path, selected_algo, selected_epoch, selected_score = get_random_top_k_model(top_num, isHousehold)

                model_path = selected_path
                # client_dir = os.path.join(ROOT_DIR, client_id)
                

                zip_filename = f"{client_id}_models.zip"
                zip_filepath = os.path.join(receive_buffer_path, zip_filename)
                
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


def get_random_models(num_household, num_government, top_k = 0):
    # 设置文件路径
    gov_log_path = os.path.join(ROOT_DIR, 'log_government.csv')
    hh_log_path = os.path.join(ROOT_DIR, 'log_household.csv')
    
    # 读取csv文件
    gov_df = pd.read_csv(gov_log_path)
    hh_df = pd.read_csv(hh_log_path)

    gov_df_sorted = gov_df.sort_values(by='score', ascending=False)
    hh_df_sorted = hh_df.sort_values(by='score', ascending=False)

    if top_k > 0:
        gov_df = gov_df_sorted.head(top_k)
        hh_df = hh_df_sorted.head(top_k)

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
    
    return selected_path, selected_algo, selected_epoch, selected_score

    

from evaluate import evaluate_policy_pools
import select
import time

# def evaluate_existing_policies(stop_event):
#     log_lock = Lock()
#     while not stop_event.is_set():
        
#         with file_lock:
            
#             print("Evaluating policy pools...")
#             print(evaluate_policy_pools(lock = log_lock))
#         time.sleep(10)
import time
import threading

def evaluate_existing_policies(stop_event, num_threads=5):
    log_lock = Lock()
    while not stop_event.is_set():
        with file_lock:
            def worker():
                
                    
                print("Evaluating policy pools...")
                print(evaluate_policy_pools(lock=log_lock))

                
            
            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            # 等待所有线程结束
            for t in threads:
                t.join()

            print('Evaluated once. Sleep for 20 seconds.')
            time.sleep(20)


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")
    stop_event = Event()
    

    # evaluationg_thread_num = 5
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     try:
            

    #         executor.submit(evaluate_existing_policies, stop_event, evaluationg_thread_num)

    #         while True:
    #             client_socket, client_address = server_socket.accept()
    #             executor.submit(handle_client, client_socket, client_address)
                
    #     except KeyboardInterrupt:
    #         print("Shutting down server...")
    #         stop_event.set()
    #         server_socket.close()
    #         # evaluate_thread.join()
    #         print("Server shutdown complete.")



    while True:
        client_socket, client_address = server_socket.accept()
        client_handler = Thread(target=handle_client, args=(client_socket, client_address))
        client_handler.start()

if __name__ == "__main__":
    main()
