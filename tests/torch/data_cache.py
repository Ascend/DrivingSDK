import hashlib
import json
import os
import pickle
import traceback
import inspect
from functools import wraps

import numpy as np
import torch


def serialize_param(param):
    # 将输入参数转字符串, 需要考虑列表或者字典的情况
    if isinstance(param, (list, tuple)):
        return ''.join(serialize_param(item) for item in param)
    elif isinstance(param, dict):
        items = sorted(param.items())
        return ''.join(serialize_param(v) for k, v in items)
    else:
        return hash_object(param) + "_"


def hash_object(obj):
    # 将对象转换为字符串
    try:
        obj_bytes = pickle.dumps(obj)
        hasher = hashlib.sha256()
        hasher.update(obj_bytes)
        hex_digest = hasher.hexdigest()[:10]
    except Exception as e:
        # 有些对象没法被sha256哈希化，暂时跳过
        hex_digest = ""
    
    return hex_digest


def save_data(data, save_path, case_name):
    file_names = []
    # 考虑有多个返回值的情况
    if isinstance(data, tuple) or isinstance(data, list):
        for i, result in enumerate(data):
            if isinstance(result, np.ndarray):
                filename_i = case_name + str(i) + ".npy"
                file_save_path = os.path.join(save_path, filename_i)
                np.save(file_save_path, result)
            elif isinstance(result, torch.Tensor):
                filename_i = case_name + str(i) + ".pth"
                file_save_path = os.path.join(save_path, filename_i)
                torch.save(result, file_save_path)
            elif isinstance(result, (int, float, list, tuple)):
                filename_i = case_name + str(i) + ".json"
                file_save_path = os.path.join(save_path, filename_i)
                result_ = {}
                result_["result"] = result
                with open(file_save_path, 'w') as json_file:
                    json.dump(result_, json_file)
            else:
                raise ValueError(f"Save cache data failed, return data type should be np.ndarray, torch.Tensor, int or float, but got {type(result)}")
            
            file_names.append(filename_i)

    elif isinstance(data, np.ndarray):
        filename = case_name + ".npy"
        file_save_path = os.path.join(save_path, filename)
        np.save(file_save_path, data)
        file_names.append(filename)
    elif isinstance(data, torch.Tensor):
        filename = case_name + ".pth"
        file_save_path = os.path.join(save_path, filename)
        torch.save(data, file_save_path)
        file_names.append(filename)
    elif isinstance(data, (int, float, list, tuple)):
        filename = case_name + ".json"
        file_save_path = os.path.join(save_path, filename)
        result_ = {}
        result_["result"] = data
        with open(file_save_path, 'w') as json_file:
            json.dump(result_, json_file)
        file_names.append(filename)
    else:
        raise ValueError(f"Save cache data failed, return data type should be np.ndarray, torch.Tensor, int or float, but got {type(data)}")
        

    return file_names


def load_data(save_path, file_names):
    results = []
    for file_name in file_names:
        file_path = os.path.join(save_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Load cache data failed. {file_path} not exists.")
        if file_name.endswith(".npy"):
            result = np.load(file_path)
            results.append(result)
        elif file_name.endswith(".pth"):
            result = torch.load(file_path)
            results.append(result)
        elif file_name.endswith(".json"):
            with open(file_path, 'r') as json_file:
                result = json.load(json_file)["result"]
            results.append(result)
            
    return results


def golden_data_cache(ut_name, save_path=None, refresh_data=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 基于args和kwargs生成缓存数据的casename, 表示Golden Function输入的参数
            case_name = ''
            for arg in args:
                case_name += (serialize_param(arg))
            for k, v in kwargs.items():
                case_name += (serialize_param(v))
                
            save_path_ = save_path
            if save_path_ is None:
                if os.getenv('MXDRIVING_CACHE_PATH', None) is not None:
                    save_path_ = os.getenv('MXDRIVING_CACHE_PATH', None)
                else:
                    current_file_path = os.path.abspath(__file__)
                    save_path_ = os.path.dirname(current_file_path)
                ut_name_ = os.path.basename(ut_name)
                ut_name_ = os.path.splitext(ut_name_)[0]
                save_path_ = os.path.join(save_path_, "data_cache", ut_name_, func.__name__) 
            cache_data_path = os.path.join(save_path_, case_name + ".json")
            
            # 如果路径下没有缓存，则重新生成缓存数据
            if not os.path.exists(cache_data_path) or refresh_data:
                if not os.path.exists(save_path_):
                    os.makedirs(save_path_)
                # 用于存储该case下面所有cache data的数据名称
                cache_data_names = {}
                
                results = func(*args, **kwargs)
                
                # 保存数据
                try:
                    file_names = save_data(results, save_path_, case_name)
                    if len(file_names) > 0:
                        cache_data_names[case_name] = file_names
                        with open(cache_data_path, 'w') as f:
                            json.dump(cache_data_names, f)
                    print(f"Cache data saved in {save_path_}.")
                except Exception as e:
                    print("Failed to save cache.")
                    traceback.print_exc()
            
            else:
                with open(cache_data_path, 'r') as file:
                    cache_data_names = json.load(file)
                file_names = cache_data_names[case_name]
                # 读取数据
                try:
                    results = load_data(save_path_, file_names)
                    if len(results) == 1:
                        results = results[0]
                    else:
                        results = tuple(results)
                    print(f"Load cache data from {save_path_}.")
                except Exception as e:
                    results = func(*args, **kwargs)
                    print("Failed to load cache, using golden function to generate data.")
                    traceback.print_exc()
                
            return results
        return wrapper
    return decorator