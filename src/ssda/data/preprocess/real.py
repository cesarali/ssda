import os
import torch
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass




def obtain_all_results_folders(all_result_parent_folder):
    path = Path(all_result_parent_folder)
    all_results_paths = []
    for path_ in path.glob("*/"):
        all_results_paths.append(path_)
    return all_results_paths

def obtain_all_results_from_path(result_path_example):
    data_dict = {}
    data_files = []
    for data_file in result_path_example.glob("*.mat"):
        data_files.append(data_file)

    for data_file in data_files:
        mat = scipy.io.loadmat(data_file, squeeze_me=True, struct_as_record=False)
        result = mat["result"]
        data_dict["{0}_{1}".format(result_path_example.name, data_file.name)] = result
    return data_dict

def create_client_dict(full_data):
    selected_client_data = {"names": [],
                            "y_diff": [],
                            "PoDm_dist": []}

    for name_, result in full_data.items():
        selected_client_data["names"].append(name_)
        data_ = result.Ydiff
        inpute_data = data_
        selected_client_data["y_diff"].append(inpute_data)
        selected_client_data["PoDm_dist"].append(result.PoDm_dist)
    return selected_client_data

def obtain_all_results(all_result_parent_folder):
    all_results_paths = obtain_all_results_folders(all_result_parent_folder)
    full_data = {}
    for result_path_example in tqdm(all_results_paths):
        full_data.update(obtain_all_results_from_path(result_path_example))
    return full_data

if __name__=="__main__":
    all_result_parent_folder = "D:/Projects/Clinical_Studies/CortBS_DEGUM_2022/06_Results/"
    full_data = obtain_all_results(all_result_parent_folder)
