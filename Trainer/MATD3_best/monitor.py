import os  
import glob  
import time  
import sys
  
def delete_oldest_files(folder_path, max_files=200):  
    """  
    删除指定文件夹下创建时间最早的文件，直到文件数量小于或等于max_files。  
    """  
    files = glob.glob(os.path.join(folder_path, '*'))  
    if len(files) > max_files:  
        # 获取文件的创建时间，并存储在字典中  
        file_times = {file: os.path.getctime(file) for file in files}  
        # 根据创建时间对文件进行排序并删除最早的文件  
        sorted_files = sorted(file_times, key=file_times.get)  
        for file in sorted_files[:len(files) - max_files]:  
            try:  
                os.remove(file)  
                print(f"Deleted file: {file}")  
            except Exception as e:  
                print(f"Error deleting file {file}: {e}")  
    else:  
        print(f"Number of files in {folder_path} is within limit ({len(files)} files).")  
  
def monitor_folder(folder_path, max_files=200, interval=60):  
    """  
    定期监控文件夹，并在文件数量超过max_files时删除最早的文件。  
    interval 参数指定检查之间的秒数。  
    """  
    try:  
        while True:  
            delete_oldest_files(folder_path, max_files)  
            print(f"Next check in {interval} seconds...")  
            time.sleep(interval)  # 等待指定的时间间隔  
    except KeyboardInterrupt:  
        print("Monitoring stopped manually.")  
  
# 使用示例：每60秒检查一次/path/to/folder下的文件数量，并保持在200以内
m_path = str(sys.argv[1])
num = int(sys.argv[2])
monitor_folder(m_path, num, 300)