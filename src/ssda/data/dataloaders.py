import os
import torch
from ssda.data.dataloaders_config import NISTLoaderConfig

from pathlib import Path
from torchvision import datasets, transforms

def get_dataset(config:NISTLoaderConfig):
    # Load MNIST dataset
    if config.data_set == "mnist":
        transform = [transforms.ToTensor()]
        if config.normalize:
            transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        transform = transforms.Compose(transform)
        train_dataset = datasets.MNIST(config.dataloader_data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(config.dataloader_data_dir, train=True, download=True, transform=transform)
    else:
        raise Exception("Data Loader Not Found!")

    return train_dataset,test_dataset

class NISTLoader:

    name_ = "NISTLoader"

    def __init__(self,config:NISTLoaderConfig):
        self.config = config

        self.batch_size = config.batch_size
        self.delete_data = config.delete_data

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_dir_file)

        train_dataset,test_dataset = get_dataset(self.config)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config.batch_size, shuffle=True)


    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader


class SemisupervisedLoader:

    name_ = "SemisupervisedLoader"

    def __init__(self, config:NISTLoaderConfig):
        self.config = config

        self.batch_size = config.batch_size
        self.delete_data = config.delete_data

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_dir_file)

        train_dataset,test_dataset = get_dataset(self.config)

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader


