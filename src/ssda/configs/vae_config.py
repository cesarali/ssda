import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from ssda.configs.files_config import ExperimentFiles
from ssda.models.encoder_config import EncoderConfig
from ssda.models.decoder_config import DecoderConfig
from ssda.trainers.vae_trainer_config import VAETrainerConfig
from ssda.data.dataloaders_config import NISTLoaderConfig
from ssda.data.porous_dataloaders_config import PorousDataLoaderConfig

all_dataloaders_configs = {"NISTLoader":NISTLoaderConfig,"PorousDataLoader":PorousDataLoaderConfig}
all_encoders_configs = {"Encoder":EncoderConfig}
all_decoders_configs = {"Decoder":DecoderConfig}
all_trainers_configs = {"VAETrainer":VAETrainerConfig}


@dataclass
class VAEConfig:

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = True
    experiment_name :str = 'vae'
    experiment_type :str = 'mnist'
    experiment_indentifier :str  = None
    init_model_path = None

    # ssda variables ------------------------------------------------------------
    z_dim: int = 20

    # all configs ---------------------------------------------------------------
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()

    dataloader: NISTLoaderConfig = NISTLoaderConfig()
    trainer: VAETrainerConfig = VAETrainerConfig()
    experiment_files:ExperimentFiles = None

    def __post_init__(self):
        if isinstance(self.experiment_files,dict):
            self.experiment_files = ExperimentFiles(**self.experiment_files)
        else:
            self.experiment_files = ExperimentFiles(delete=self.delete,
                                                    experiment_name=self.experiment_name,
                                                    experiment_type=self.experiment_type,
                                                    experiment_indentifier=self.experiment_indentifier)

        if isinstance(self.encoder, dict):
            self.encoder = all_encoders_configs[self.encoder["name"]](**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = all_decoders_configs[self.decoder["name"]](**self.decoder)
        if isinstance(self.dataloader,dict):
            self.dataloader = all_dataloaders_configs[self.dataloader["name"]](**self.dataloader)
        if isinstance(self.trainer,dict):
            self.trainer = all_trainers_configs[self.trainer["name"]](**self.trainer)

    def initialize_new_experiment(self,
                                  experiment_name: str = None,
                                  experiment_type: str = None,
                                  experiment_indentifier: str = None):
        if experiment_name is not None:
            self.experiment_name = experiment_name
        if experiment_type is not None:
            self.experiment_type = experiment_type
        if experiment_indentifier is not None:
            self.experiment_indentifier = experiment_indentifier

        self.align_configurations()
        self.experiment_files.create_directories()
        self.config_path = self.experiment_files.config_path
        self.save_config()

    def align_configurations(self):
        pass
        #self.encoder.input_size = self.dataloader.dimensions_per_variable * self.dataloader.number_of_variables

    def save_config(self):
        config_as_dict = asdict(self)
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)

def get_config_from_file(experiment_name, experiment_type, experiment_indentifier) -> VAEConfig:
    from ssda import results_path

    experiment_dir = os.path.join(results_path, experiment_name)
    experiment_type_dir = os.path.join(experiment_dir, experiment_type)
    results_dir = os.path.join(experiment_type_dir, experiment_indentifier)
    results_dir_path = Path(results_dir)
    if results_dir_path.exists():
        config_path = os.path.join(results_dir, "config.json")
        config_path_json = json.load(open(config_path, "r"))
        config_path_json["delete"] = False
        config = VAEConfig(**config_path_json)
        return config
    else:
        raise Exception("Folder Does Not Exist")
