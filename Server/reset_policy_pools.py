import os
import shutil

def clear_directory(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
                print(f'Removed file: {file_path}')
            for name in dirs:
                dir_path = os.path.join(root, name)
                shutil.rmtree(dir_path)
                print(f'Removed directory: {dir_path}')
        print(f'All files and directories under {path} have been removed.')
    else:
        print(f'The path {path} does not exist.')

# 指定路径
directory_path = 'Server/policy_pools'
clear_directory(directory_path)
