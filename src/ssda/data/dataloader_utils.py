from typing import Union

from ssda.configs.vae_config import VAEConfig
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.data.dataloaders_config import NISTLoaderConfig
from ssda.data.dataloaders_config import SemisupervisedLoaderConfig
from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig

from ssda.data.dataloaders import NISTLoader
from ssda.data.dataloaders import SemisupervisedLoader
from ssda.data.porous_dataloaders import SemisupervisedPorousLoader

def load_dataloader(config:Union[VAEConfig,NISTLoaderConfig,SemisupervisedLoaderConfig]):
    config_:NISTLoaderConfig

    if isinstance(config,VAEConfig):
        config_ = config.dataloader
    elif isinstance(config,SSVAEConfig):
        config_ = config.dataloader
    elif isinstance(config, NISTLoaderConfig):
        config_ = config
    elif isinstance(config, SemisupervisedLoaderConfig):
        config_ = config
    elif isinstance(config, SemisupervisedLoaderConfig):
        config_ = config
    else:
        raise Exception("Config Does Not Exist")

    if config_.name == "NISTLoader":
        dataloader = NISTLoader(config_)
    elif config_.name == "SemisupervisedLoader":
        dataloader = SemisupervisedLoader(config_)
    elif config_.name == "SemisupervisedPorousLoader":
        dataloader = SemisupervisedPorousLoader(config_)
        SemisupervisedPorousLoader
    else:
        raise Exception("Dataloader Does Not Exist")

    return dataloader

