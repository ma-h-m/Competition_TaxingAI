import os
import glob
import shutil
# 定义需要清理的目录

def clear_certain_pool(models_directory, csv_directory):
# 清除 models_directory 下的所有文件
    for filename in os.listdir(models_directory):
        file_path = os.path.join(models_directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # 删除文件
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除文件夹及其内容
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # 清除 csv_directory 下的所有 .csv 文件
    csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))
    for csv_file in csv_files:
        try:
            os.remove(csv_file)
        except Exception as e:
            print(f'Failed to delete {csv_file}. Reason: {e}')

    print("Cleanup complete.")

models_directory = 'TaxAI_modified/agents/model_pools/long_term/models'
csv_directory = 'TaxAI_modified/agents/model_pools/long_term'

clear_certain_pool(models_directory, csv_directory)
# 清除 long_term 目录下的所有文件
models_directory = 'TaxAI_modified/agents/model_pools/short_term/models'
csv_directory = 'TaxAI_modified/agents/model_pools/short_term'

clear_certain_pool(models_directory, csv_directory)
# 清除 short_term 目录下的所有文件
models_directory = 'TaxAI_modified/agents/model_pools/top_k/models'
csv_directory = 'TaxAI_modified/agents/model_pools/top_k'

clear_certain_pool(models_directory, csv_directory)

models_directory = 'TaxAI_modified/agents/model_pools/models_from_server/models'
csv_directory = 'TaxAI_modified/agents/model_pools/models_from_server'

clear_certain_pool(models_directory, csv_directory)
