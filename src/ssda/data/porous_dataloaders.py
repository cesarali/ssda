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

from torch import nn
from ssda import data_path
from torchvision import transforms
from ssda.data.base_dataloaders import TransformsDictDataSet
from torch.utils.data import DataLoader, TensorDataset
from ssda.data.transforms import UnsqueezeTensorTransform
from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig
from ssda.configs.ssvae_porous_config import SSPorousVAEConfig
from dataclasses import dataclass
from ssda.data.porous_dataloaders_config import PorousDataLoaderConfig


def get_transforms(config:PorousDataLoaderConfig,type="patients")->Tuple[str,transforms.Compose]:
    """
    :param config:
    :param type:
    :return:
    """
    if type == "patients":
        transforms_ = transforms.Compose([
            UnsqueezeTensorTransform(),
            #transforms.Normalize(mean=[patient_metadata["rescaling"][0]], std=[patient_metadata["rescaling"][0]]),
            transforms.Resize((11, 24))
        ])
        key_of_transforms = "y_diff"
        return key_of_transforms,transforms_
    elif type == "simulations":

        transforms_ = transforms.Compose([
            UnsqueezeTensorTransform(),
            #transforms.Normalize(mean=[simulations_metadata["rescaling"][0]],std=[simulations_metadata["rescaling"][0]]),
            transforms.Resize((11, 24))
        ])
        key_of_transforms = "nds"
        return key_of_transforms,transforms_

def get_datasets(config:PorousDataLoaderConfig,type="patients"):
    """
    :param config:
    :param type:
    :return:
    """
    from ssda import data_path

    if type == "patients":
        from ssda.data.preprocess.real import obtain_all_results
        real_filtered_path = os.path.join(data_path, "preprocessed", "filtered_real_images.pkl")
        final_reduced_real_image = pickle.load(open(real_filtered_path, "rb"))

        #all_result_parent_folder = "D:/Projects/Clinical_Studies/CortBS_DEGUM_2022/06_Results/"
        #full_data = obtain_all_results(all_result_parent_folder)

        #real_dz = mat_result.dz
        #real_frequencies = mat_result.f_sampling
        #real_image = mat_result.Ydiff

        dataset_dict = {"images": [image for k, image in final_reduced_real_image.items()]}
        return dataset_dict

    elif type == "simulations":

        simulation_filtered_path = os.path.join(data_path, "preprocessed", "filtered_simulation_images.pkl")
        final_reduced_simulation_image = pickle.load(open(simulation_filtered_path, "rb"))

        raw_path = os.path.join(data_path, "raw")
        data_path = Path(raw_path)
        mat_path = list(data_path.glob("*.mat"))[0]
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        dataset_dict = {}
        dataset_dict["images"] = [image for k,image in final_reduced_simulation_image.items()]
        for target_string in config.target_for_simulation:
            dataset_dict[target_string] = mat[target_string]

        image_example = dataset_dict["images"][0]
        config.input_dim = image_example.shape[0]* image_example.shape[1]

        return dataset_dict


class PorousDataLoaderDict:

    name_ = "porous_data_loader"

    def __init__(self,
                 config:PorousDataLoaderConfig,
                 type=None):
        super(PorousDataLoaderDict, self).__init__()
        self.training_proportion = config.training_proportion
        self.batch_size = config.batch_size
        if type is None:
            self.type = config.type
        else:
            self.type = type
        dict_datasets = get_datasets(config,self.type)
        #self.key_of_transforms, self.transforms_ = get_transforms(config,type)
        self.key_of_transforms, self.transforms_ = None,None
        self.define_dataset_and_dataloaders(dict_datasets)
        self.config = config


    def define_dataset_and_dataloaders(self, dict_datasets):
        if isinstance(dict_datasets, torch.Tensor):
            dataset = TensorDataset(dict_datasets)
        elif isinstance(dict_datasets, dict):
            dataset = TransformsDictDataSet(dict_datasets, self.transforms_, self.key_of_transforms)

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

    def train(self,data_type="label"):
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
    from pprint import pprint
    from dataclasses import asdict

    config = SSPorousVAEConfig()
    config.dataloader.batch_size = 2

    dataloader = PorousDataLoaderDict(config,"patients")
    databatch = next(dataloader.train().__iter__())
    print(databatch)

    dataloader = PorousDataLoaderDict(config,"simulations")
    databatch = next(dataloader.train().__iter__())
    print(databatch)