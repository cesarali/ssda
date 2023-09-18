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

    def __post_init__(self):
        from ssda import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw")
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+".tr")


@dataclass
class SemisupervisedConfig:
    name:str = "SemisupervisedLoader"
    data_set:str = "mnist"
    dataloader_data_dir:str = None

    normalize:bool = True
    input_dim: int = 784
    batch_size: int = 32
    labeled_proportion: float = 0.2

    def __post_init__(self):
        from ssda import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw")
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+"_ss.tr")