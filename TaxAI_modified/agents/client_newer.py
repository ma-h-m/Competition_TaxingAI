import os
import paramiko
from scp import SCPClient

# SSH 连接信息
SERVER_IP = '127.0.0.1'
SERVER_PORT = 2222  # 通常 SSH 使用端口 22
USERNAME = 'your_username'
PASSWORD = 'your_password'

# 客户端唯一 ID
CLIENT_ID = 'unique_client_id'

# 本地同步文件夹
LOCAL_DIR = '/home/mhm/workspace/Competition_TaxingAI/TaxAI_modified/agents/model_pools'
REMOTE_DIR = 'Server/policy_pools'

def setup_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER_IP, port=SERVER_PORT, username=USERNAME, password=PASSWORD)
    return ssh

def send_client_id(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command(CLIENT_ID)
    stdout.channel.recv_exit_status()

def upload_file(ssh_client, local_path, remote_path):
    scp = SCPClient(ssh_client.get_transport())
    scp.put(local_path, remote_path)
    print(f'Uploaded: {local_path} to {remote_path}')

def download_file(ssh_client, remote_path, local_path):
    scp = SCPClient(ssh_client.get_transport())
    scp.get(remote_path, local_path)
    print(f'Downloaded: {remote_path} to {local_path}')

def sync_to_server(remote_dir):
    ssh_client = setup_ssh()

    # 传输客户端 ID，并在服务器上创建相应的文件夹
    send_client_id(ssh_client)
    
    for root, dirs, files in os.walk(LOCAL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, LOCAL_DIR)
            remote_path = os.path.join(remote_dir, relative_path)
            
            upload_file(ssh_client, local_path, remote_path)
    
    ssh_client.close()

def sync_from_server(remote_dir):
    ssh_client = setup_ssh()

    # 传输客户端 ID
    send_client_id(ssh_client)
    
    for root, dirs, files in os.walk(REMOTE_DIR):
        for file in files:
            remote_path = os.path.join(root, file)
            relative_path = os.path.relpath(remote_path, REMOTE_DIR)
            local_path = os.path.join(LOCAL_DIR, relative_path)
            
            download_file(ssh_client, remote_path, local_path)
    
    ssh_client.close()

if __name__ == "__main__":
    # remote_dir = input("请输入服务器上的远程目录: ")
    remote_dir = REMOTE_DIR

    sync_to_server(remote_dir)
    # action = input("输入 'upload' 将本地文件同步到服务器, 输入 'download' 将服务器文件同步到本地: ")
    # if action == 'upload':
        # sync_to_server(remote_dir)
    # elif action == 'download':
        # sync_from_server(remote_dir)
    # else:
    #     print("无效的输入，请输入 'upload' 或 'download'")
