import socket
import os

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
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, folder_path)
            push_file_with_path(filepath, relative_path)

def push_file_with_path(filepath, relative_path):
    filesize = os.path.getsize(filepath)
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"PUSH{SEPARATOR}{relative_path}{SEPARATOR}{filesize}{SEPARATOR}".encode())

    with open(filepath, "rb") as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            client_socket.sendall(bytes_read)
    
    client_socket.send(f"DONE{SEPARATOR}{relative_path}".encode())
    client_socket.close()

def fetch_file(filename, save_path):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    client_socket.send(f"FETCH{SEPARATOR}{filename}".encode())
    
    received = client_socket.recv(BUFFER_SIZE).decode()
    if received == "FILE_NOT_FOUND":
        print(f"File {filename} not found on server.")
    else:
        filesize = int(received.split(SEPARATOR)[0])
        with open(save_path, "wb") as f:
            bytes_received = 0
            while bytes_received < filesize:
                bytes_read = client_socket.recv(BUFFER_SIZE)
                if not bytes_read:
                    break
                f.write(bytes_read)
                bytes_received += len(bytes_read)
    
    client_socket.close()

def initial_communicate_with_server(client_id="Unknown"):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    client_socket.send(f"INIT{SEPARATOR}{client_id}".encode())
    response = client_socket.recv(BUFFER_SIZE).decode()
    print(response)
    client_socket.close()

if __name__ == "__main__":
    # 示例用法
    initial_communicate_with_server("Client123")
    push_folder("/home/mhm/workspace/Competition_TaxingAI/TaxAI_modified/agents/model_pools")
    # fetch_file("file_to_fetch", "save_path")
