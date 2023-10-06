import torch

def standardize_tensor(tensor):
    tensor_mean = torch.mean(tensor)
    tensor_std = torch.std(tensor)
    standardized_tensor = (tensor - tensor_mean) / tensor_std
    return standardized_tensor

def normalize_tensor(tensor):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor