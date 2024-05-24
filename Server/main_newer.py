import os
import paramiko
import threading
import socket
from scp import SCPClient

# SSH 连接信息
SERVER_IP = '0.0.0.0'
SERVER_PORT = 2222  # 通常 SSH 使用端口 22
USERNAME = 'your_username'
PASSWORD = 'your_password'
BASE_DIR = 'Server/policy_pools'

class MyServer(paramiko.ServerInterface):
    def __init__(self, sock):
        self.event = threading.Event()
        self.sock = sock

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_password(self, username, password):
        if username == USERNAME and password == PASSWORD:
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_exec_request(self, channel, command):
        global CLIENT_ID
        CLIENT_ID = str(command.strip())
        channel.send_exit_status(0)
        self.event.set()
        return True

def handle_client(sock):
    server = MyServer(sock)

    # 绑定并监听端口
    server_sock = paramiko.Transport(sock)
    server_sock.add_server_key(paramiko.RSAKey(filename='server_rsa.key'))

    server_sock.start_server(server=server)
    channel = server_sock.accept(20)
    if channel is None:
        print("连接失败")
        return

    server.event.wait(10)
    if CLIENT_ID:
        print(f"客户端 ID: {CLIENT_ID}")
        client_dir = os.path.join(BASE_DIR, CLIENT_ID)
        os.makedirs(client_dir, exist_ok=True)
        # remote_dir = input(f"请输入服务器上的远程目录: ")
        # action = input(f"输入 'upload' 将本地文件同步到客户端({CLIENT_ID}), 输入 'download' 将客户端({CLIENT_ID})文件同步到本地: ")
        # if action == 'upload':
        #     # Use SCP to upload files to the server
        #     with paramiko.SSHClient() as ssh:
        #         ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #         ssh.connect(hostname=SERVER_IP, port=SERVER_PORT, username=USERNAME, password=PASSWORD)
        #         with SCPClient(ssh.get_transport()) as scp:
        #             scp.put(local_dir, recursive=True, remote_path=remote_dir)

    channel.close()
    server_sock.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((SERVER_IP, SERVER_PORT))
    server.listen(5)
    
    while True:
        client_sock, client_addr = server.accept()
        print(f'新客户端连接: {client_addr}')
        
        client_thread = threading.Thread(target=handle_client, args=(client_sock,))
        client_thread.start()

if __name__ == "__main__":
    main()
