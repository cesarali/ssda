import os
from typing import List
from dataclasses import dataclass
from dataclasses import dataclass,fields,field

@dataclass
class PorousDataLoaderConfig:
    name:str = "PorousDataLoader"
    dataloader_data_dir:str = None
    delete:bool = False
    normalize:bool = True

    type:str = "patients" #patients,simulations
    input_size:int = None
    batch_size: int = 32
    training_proportion:float = 0.8
    number_of_labels: int = 1

    target_name:str = "PoDm_mean" #"PoDm_mean", "PoDm_std", "PoDm_q90", "PoDm_q10", "absorption"

    target_for_patient:List = field(default_factory=lambda:["y_diff"])
    target_for_simulation:List = field(default_factory=lambda:["PoDm_mean","PoDm_std"])


@dataclass
class SemisupervisedLoaderPorousConfig:
    name:str = "SemisupervisedPorousLoader"
    dataloader_data_dir:str = None
    delete:bool = False
    normalize:bool = True

    input_size:int = None
    batch_size: int = 32
    training_proportion:float = 0.8
    number_of_labels: int = 1

    target_name:str = "PoDm_mean" #"PoDm_mean", "PoDm_std", "PoDm_q90", "PoDm_q10", "absorption"

    target_for_patient:List = field(default_factory=lambda:["y_diff"])
    target_for_simulation:List = field(default_factory=lambda:["PoDm_mean","PoDm_std"])

    def __post_init__(self):
        from ssda import data_path as data_dir
        self.dataloader_data_dir = os.path.join(data_dir, "raw", "porous")
        self.number_of_labels = len(self.target_for_simulation)