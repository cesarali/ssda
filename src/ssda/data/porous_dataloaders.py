import os
import scipy
import torch
import pickle
from abc import ABC
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset,DataLoader,random_split
from typing import Optional, Union, Tuple
from torchvision import transforms

from ssda import data_path
from ssda.data.base_dataloaders import TransformsDictDataSet
from torch.utils.data import DataLoader, TensorDataset

class UnsqueezeTensorTransform:

    def __init__(self,axis=0):
        self.axis = axis
    def __call__(self, tensor:torch.Tensor):
        return tensor.unsqueeze(self.axis)

class PorousDataLoaderDict(ABC):
    name_ = "porous_data_loader"

    def __init__(self,
                 X: Union[torch.Tensor, dict] = None,
                 type="client",
                 batch_size: int = 32,
                 training_proportion: float = 0.9,
                 device: torch.device = torch.device("cpu"),
                 rank: int = 0,
                 in_house=False,
                 **kwargs):
        super(PorousDataLoaderDict, self).__init__()
        self.training_proportion = training_proportion
        self.batch_size = batch_size

        if type == "client":
            patient_dir = os.path.join(data_path, "raw", "porous", "patient")
            patient_metadata_path = os.path.join(patient_dir, "client_metadata.cp")

            with open(patient_metadata_path, "rb") as file:
                patient_metadata = pickle.load(file)

            self.transforms = transforms.Compose([
                UnsqueezeTensorTransform(),
                transforms.Normalize(mean=[patient_metadata["rescaling"][0]], std=[patient_metadata["rescaling"][0]]),
                transforms.Resize((11, 24))
            ])
            self.key_of_transforms = "y_diff"

        elif type == "simulations":
            simulations_dir = os.path.join(data_path, "raw", "porous", "simulation")
            simulations_metadata_path = os.path.join(simulations_dir, "simulations_metadata.cp")

            with open(simulations_metadata_path, "rb") as file:
                simulations_metadata = pickle.load(file)

            self.transforms = transforms.Compose([
                UnsqueezeTensorTransform(),
                transforms.Normalize(mean=[simulations_metadata["rescaling"][0]],
                                     std=[simulations_metadata["rescaling"][0]]),
                transforms.Resize((11, 24))
            ])
            self.key_of_transforms = "nds"

        self.define_dataset_and_dataloaders(X)

    def define_dataset_and_dataloaders(self, X, training_proportion=None, batch_size=None):
        if training_proportion is not None:
            self.training_proportion = training_proportion
        if batch_size is not None:
            self.batch_size = batch_size

        if isinstance(X, torch.Tensor):
            dataset = TensorDataset(X)
        elif isinstance(X, dict):
            dataset = TransformsDictDataSet(X, self.transforms, self.key_of_transforms)

        self.total_data_size = len(dataset)
        self.training_data_size = int(self.training_proportion * self.total_data_size)
        self.test_data_size = self.total_data_size - self.training_data_size
        training_dataset, test_dataset = random_split(dataset, [self.training_data_size, self.test_data_size])

        self._train_iter = DataLoader(training_dataset, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, batch_size=self.batch_size)

    def train(self):
        return self._train_iter

    def test(self):
        return self._test_iter


from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig

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


def get_simulations(config:SemisupervisedLoaderPorousConfig):
    from ssda.data.preprocess.porous import pick_data
    from ssda.data.preprocess.porous import inpute_data_cut
    from ssda import data_path as data_dir

    data_dir = os.path.join(data_dir, "raw", "porous")
    data_path = os.path.join(data_dir,"Charite_CortBS_Simulations.mat")
    mat = scipy.io.loadmat(data_path)

    X_,Y = pick_data(mat,target_name=config.target_name)
    X_ = X_.transpose(2,0,1)
    X_reduced = torch.Tensor(inpute_data_cut(X_,cut_size_x=config.cut_size_x,cut_size_y = config.cut_size_y))
    Y = torch.Tensor(Y)
    config.input_dim = X_reduced.shape[1]*X_reduced.shape[2]

    X_reduced = normalize_tensor(X_reduced)
    Y = normalize_tensor(Y)

    assert not torch.isnan(torch.Tensor(X_reduced)).any()

    mydataset = TensorDataset(X_reduced, Y)

    return mydataset


class SemisupervisedPorousLoader:

    name_ = "SemisupervisedPorousLoader"

    def __init__(self, config:SemisupervisedLoaderPorousConfig):
        self.config = config

        self.batch_size = config.batch_size
        self.training_proportion = config.training_proportion
        self.delete_data = config.delete

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_dir_file)

        label_dataset = get_simulations(self.config)
        unlabel_dataset = get_simulations(self.config)

        self.total_data_size = len(label_dataset)
        self.training_data_size = int(self.training_proportion * self.total_data_size)
        self.test_data_size = self.total_data_size - self.training_data_size
        train_label_dataset, test_label_dataset = random_split(label_dataset, [self.training_data_size, self.test_data_size])

        self.train_labeled_loader = DataLoader(train_label_dataset, batch_size=self.batch_size)
        self.test_labeled_loader = DataLoader(test_label_dataset, batch_size=self.batch_size)
        self.train_unlabeled_loader = DataLoader(unlabel_dataset, batch_size=self.batch_size)


    def train(self, data_type="label"):
        if data_type == "label":
            return self.train_labeled_loader
        else:
            return self.train_unlabeled_loader

    def test(self,data_type="label"):
        if data_type == "label":
            return self.test_labeled_loader
        else:
            return None


if __name__=="__main__":

    dataset = get_simulations()

    train_dataloader = DataLoader(dataset,batch_size=23)
    databatch = next(train_dataloader.__iter__())

    print(databatch[0].shape)
    print(databatch[1].shape)
