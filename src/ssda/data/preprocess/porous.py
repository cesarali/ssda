import numpy as np
import scipy
import torch

def pick_data(mat,target_name="PoDm_mean"):
    X_ = mat['NDS']
    Y = mat[target_name].T
    return  X_,Y

def inpute_data_cut(X_,cut_size_x = 5,cut_size_y = 5):
    # Calculate the mean of each column
    X_reduced = X_[:,cut_size_x:-cut_size_x,cut_size_y:-cut_size_y]
    assert not np.isnan(X_reduced).any()
    return X_reduced

def inpute_data_mean(X_):
    # Calculate the mean of each column
    col_mean = np.nanmean(X_, axis=0)
    row_mean = np.nanmean(X_, axis=1)
    # Replace NaN values with the mean of the column
    X_[np.isnan(X_)] = np.take(col_mean, np.isnan(X_).nonzero()[1])
    X_[np.isnan(X_)] = np.take(row_mean, np.isnan(X_).nonzero()[1])
    return X_

def normalize(X_reduced,YF):
    # Calculate mean and standard deviation of the input X
    X_mean = np.mean(X_reduced, axis=0)
    X_std = np.std(X_reduced, axis=0)
    # Normalize the input X
    X_normalized = (X_reduced - X_mean) / X_std
    # Calculate mean and standard deviation of the output Y
    Y_mean = np.mean(YF, axis=0)
    Y_std = np.std(YF, axis=0)
    # Normalize the output Y
    Y_normalized = (YF - Y_mean) / Y_std
    return X_normalized, Y_normalized

def permute_data(X,Y):
    permutation = torch.randperm(torch.Tensor(X).size()[0])
    X = torch.Tensor(X)[permutation].numpy()
    Y = torch.Tensor(Y)[permutation].numpy()
    return X,Y


def inpute_nans(data_):
    row_, cols_ = np.where(~indices)
    if len(data_.shape) == 2:
        j_start, j_end = min(cols_), max(cols_)
        i_start, i_end = min(row_), max(row_)
        inpute_data = data_[i_start:i_end, j_start:j_end]
        return inpute_data
    elif len(data_.shape) == 1:
        i_start, i_end = min(row_), max(row_)
        inpute_data = data_[i_start:i_end, j_start:j_end]
        return inpute_data


def obtain_all_results(result_path_example):
    data_dict = {}
    data_files = []
    for data_file in result_path_example.glob("*.mat"):
        data_files.append(data_file)

    for data_file in data_files:
        mat = scipy.io.loadmat(data_file, squeeze_me=True, struct_as_record=False)
        result = mat["result"]
        data_dict["{0}_{1}".format(result_path_example.name, data_file.name)] = result

    return data_dict