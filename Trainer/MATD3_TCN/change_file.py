import os  
import sys
  
def rename_files_in_folder(folder_path):  
    for filename in os.listdir(folder_path):  
        file_path = os.path.join(folder_path, filename)  
        # 检查是否为文件  
        if os.path.isfile(file_path):  
            # 分离文件名和扩展名  
            base_name, extension = os.path.splitext(filename)  
            # 构造新的文件名，保持扩展名不变  
            new_filename = base_name + "_a40" + extension  
            # 重命名文件  
            new_file_path = os.path.join(folder_path, new_filename)  
            os.rename(file_path, new_file_path)  
  
# 替换为您要处理的文件夹的路径  
folder_path = str(sys.argv[1])
rename_files_in_folder(folder_path)