import socket
import os
import zipfile

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5004
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
USER_ID = "test2"

def push_file(filepath):
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"PUSH{SEPARATOR}{filename}{SEPARATOR}{filesize}{SEPARATOR}".encode())
    
    # 设置超时时间为10秒
    client_socket.settimeout(10)

    try:
        # 等待服务器的回应
        response = client_socket.recv(BUFFER_SIZE).decode()
        if response != "READY":
            print(f"Server is not ready, received: {response}")
            client_socket.close()
            return
    except socket.timeout:
        print("Server response timeout.")
        client_socket.close()
        return
    
    with open(filepath, "rb") as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            client_socket.sendall(bytes_read)
    
    response = client_socket.recv(BUFFER_SIZE).decode()
    print(f"Server response: {response}")

    client_socket.close()

def push_folder(folder_path, user_id=USER_ID):
    existing_files = get_existing_files(user_id)
    zip_filename = f"{os.path.basename(folder_path)}.zip"
    zip_filepath = os.path.join(os.path.dirname(folder_path), zip_filename)
    
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, folder_path)
                
                # 检查文件是否已经存在于服务器上
                if file.endswith(".csv") or relative_path not in existing_files:
                    zipf.write(filepath, relative_path)
                else:
                    print(f"File {relative_path} already exists on the server. Skipping...")
    
    push_file(zip_filepath)
    os.remove(zip_filepath)

def initial_communicate_with_server(id="Unknown"):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"INIT{SEPARATOR}{id}{SEPARATOR}".encode())
    response = client_socket.recv(BUFFER_SIZE).decode()
    print(f"Server response: {response}")
    
    client_socket.close()

def get_existing_files(client_id):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"LIST{SEPARATOR}{client_id}{SEPARATOR}".encode())
    response = client_socket.recv(BUFFER_SIZE).decode()
    
    existing_files = response.split(SEPARATOR) if response else []
    if existing_files == ["$EMPTY"]:
        existing_files = []
    client_socket.close()
    
    return existing_files

def fetch_random_models(gov_model_num = 1, household_model_num = 4, user_id=USER_ID, dest_dir="TaxAI_modified/agents/model_pools/models_from_server"):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"FETCH_RANDOM_MODELS{SEPARATOR}{household_model_num}{SEPARATOR}{gov_model_num}{SEPARATOR}{USER_ID}{SEPARATOR}".encode())
    
    response = client_socket.recv(BUFFER_SIZE).decode()
    if response.startswith("FILE_NOT_FOUND"):
        print("Server did not find the requested number of models.")
        client_socket.close()
        return
    
    filesize = int(response.split(SEPARATOR)[0])
    
    # 发送准备接收文件的确认
    client_socket.send("READY".encode())
    
    # 保存文件
    zip_filename = f"{user_id}_models.zip"
    with open(zip_filename, "wb") as f:
        bytes_received = 0
        while bytes_received < filesize:
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:
                break
            f.write(bytes_read)
            bytes_received += len(bytes_read)
    
    print(f"Received {zip_filename} from server.")

    # 解压文件到指定目录
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    
    os.remove(zip_filename)
    print(f"Extracted models to {dest_dir}")
    
    client_socket.close()


if __name__ == "__main__":
    # 示例用法
    initial_communicate_with_server(USER_ID)
    # push_folder("/home/mhm/workspace/Competition_TaxingAI/TaxAI_modified/agents/model_pools")

    fetch_random_models(user_id=USER_ID)