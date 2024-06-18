import os
import shutil

def remove_all_contents(directory):
    """
    删除指定目录下的所有文件和文件夹。
    
    :param directory: 需要删除内容的目录路径
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'删除 {file_path} 时出错: {e}')
    else:
        print(f'目录 {directory} 不存在')

directories = [
    'Server/policy_pools',
    'TaxAI_modified/agents/model_pools/long_term',
    'TaxAI_modified/agents/model_pools/short_term',
    'TaxAI_modified/agents/model_pools/top_k',
    'TaxAI_3rd_version/agents/model_pools/long_term',
    'TaxAI_3rd_version/agents/model_pools/short_term',
]

for directory in directories:
    remove_all_contents(directory)
    print(f'{directory} 目录下的所有内容已删除')
