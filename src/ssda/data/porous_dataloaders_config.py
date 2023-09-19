import os
from dataclasses import dataclass

@dataclass
class SemisupervisedLoaderPorousConfig:
    name:str = "SemisupervisedPorousLoader"
    dataloader_data_dir:str = None

    delete:bool = False
    normalize:bool = True

    input_size:int = None
    batch_size: int = 32
    cut_size_x:int = 5
    cut_size_y:int = 5
    training_proportion:float = 0.8
    number_of_labels: int = 1

    target_name:str = "PoDm_mean" #"PoDm_mean", "PoDm_std", "PoDm_q90", "PoDm_q10", "absorption"

    def __post_init__(self):
        from ssda import data_path as data_dir
        self.dataloader_data_dir = os.path.join(data_dir, "raw", "porous")
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,"simulation_metadata_ss.tr")
        self.number_of_labels = 1