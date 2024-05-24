import socket
import os
from threading import Thread
import zipfile

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5001
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

# 创建服务器的根文件夹
ROOT_DIR = "Server/policy_pools"
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

ip_to_id = {}

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
                if len(args) >= 3:
                    relative_path = args[0]
                    filesize = int(args[1])
                    zip_path = os.path.join(client_dir, relative_path)
                    
                    # 确保文件夹存在
                    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
                    
                    with open(zip_path, "wb") as f:
                        bytes_received = 0
                        while bytes_received < filesize:
                            bytes_read = client_socket.recv(BUFFER_SIZE)
                            if not bytes_read:
                                break
                            f.write(bytes_read)
                            bytes_received += len(bytes_read)
                    print(f"[+] File {relative_path} received from {client_id} at ip address{client_address}.")

                    # 解压缩接收到的zip文件
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(client_dir)
                    
                    # 删除接收到的zip文件
                    os.remove(zip_path)
                    
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
        
        except ConnectionResetError:
            break
    
    client_socket.close()
    print(f"[-] {client_address} disconnected.")

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
