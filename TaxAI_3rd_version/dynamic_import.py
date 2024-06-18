import os
import importlib.util
import sys

def dynamic_import_class(module_name, file_path, class_name):
    # 获取文件所在目录的父目录的父目录，即顶级包目录
    module_dir = os.path.dirname(file_path)
    last_dir = os.path.basename(module_dir)
    top_level_package_dir = os.path.abspath(os.path.join(module_dir, os.pardir, os.pardir, os.pardir))
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    parts = module_dir.split('/')
    parts = parts[1:]
    tmp_path = '.'.join(parts)
    # 设置父包，以支持相对导入
    # setattr(module, '__package__', 'policy_pools.test1.long_term.models.' + last_dir)
    setattr(module, '__package__', tmp_path)
    
    # 将顶级包目录添加到 sys.path
    sys.path.insert(0, top_level_package_dir)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取指定类
    cls = getattr(module, class_name)
    
    # 移除顶级包目录
    sys.path.pop(0)
    
    return cls

