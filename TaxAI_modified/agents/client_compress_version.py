import socket
import os
import zipfile

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5001
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

def push_file(filepath):
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"PUSH{SEPARATOR}{filename}{SEPARATOR}{filesize}{SEPARATOR}".encode())
    
    with open(filepath, "rb") as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            client_socket.sendall(bytes_read)
    
    client_socket.send(f"DONE{SEPARATOR}{filename}".encode())
    client_socket.close()

def push_folder(folder_path):
    zip_filename = f"{os.path.basename(folder_path)}.zip"
    zip_filepath = os.path.join(os.path.dirname(folder_path), zip_filename)
    
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, folder_path)
                zipf.write(filepath, relative_path)
    
    push_file(zip_filepath)

def initial_communicate_with_server(id="Unknown"):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"INIT{SEPARATOR}{id}{SEPARATOR}".encode())
    response = client_socket.recv(BUFFER_SIZE).decode()
    print(f"Server response: {response}")
    
    client_socket.close()

if __name__ == "__main__":
    # 示例用法
    initial_communicate_with_server("test")
    push_folder("/home/mhm/workspace/Competition_TaxingAI/TaxAI_modified/agents/model_pools")
