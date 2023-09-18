import os
from dataclasses import dataclass

@dataclass
class NISTLoaderConfig:
    name:str = "NISTLoader"
    data_set:str = "mnist"
    dataloader_data_dir:str = None

    normalize = True
    input_dim: int = 784
    batch_size: int = 32
    delete_data:bool = False
    number_of_labels = 10

    def __post_init__(self):
        from ssda import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw")
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+".tr")
        if self.data_set == "mnist":
            self.number_of_labels = 10


@dataclass
class SemisupervisedLoaderConfig:
    name:str = "SemisupervisedLoader"
    data_set:str = "mnist"
    dataloader_data_dir:str = None

    delete:bool = False
    normalize:bool = True
    input_dim: int = 784
    batch_size: int = 32
    labeled_proportion: float = 0.2
    number_of_labels: int = 10

    def __post_init__(self):
        from ssda import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw")
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+"_ss.tr")
        if self.data_set == "mnist":
            self.number_of_labels = 10