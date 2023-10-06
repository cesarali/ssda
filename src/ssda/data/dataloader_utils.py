from typing import Union

from ssda.configs.vae_config import VAEConfig
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.data.dataloaders_config import NISTLoaderConfig
from ssda.data.dataloaders_config import SemisupervisedLoaderConfig
from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig,PorousDataLoaderConfig

from ssda.data.dataloaders import NISTLoader
from ssda.data.dataloaders import SemisupervisedLoader
from ssda.data.porous_dataloaders import SemisupervisedPorousLoader
from ssda.data.porous_dataloaders import PorousDataLoaderDict

def load_dataloader(config:Union[VAEConfig,NISTLoaderConfig,SemisupervisedLoaderPorousConfig,SemisupervisedLoaderConfig,PorousDataLoaderConfig],type=None):

    if isinstance(config,VAEConfig):
        config_ = config.dataloader
    elif isinstance(config,SSVAEConfig):
        config_ = config.dataloader
    elif isinstance(config, (NISTLoaderConfig, SemisupervisedLoaderConfig, SemisupervisedLoaderPorousConfig, PorousDataLoaderConfig)):
        config_ = config

    else:
        raise Exception("Config Does Not Exist")

    if config_.name == "NISTLoader":
        dataloader = NISTLoader(config_)
    elif config_.name == "SemisupervisedLoader":
        dataloader = SemisupervisedLoader(config_)
    elif config_.name == "SemisupervisedPorousLoader":
        dataloader = SemisupervisedPorousLoader(config_)
    elif isinstance(config_,PorousDataLoaderConfig):
        dataloader = PorousDataLoaderDict(config_, type)
    else:
        raise Exception("Dataloader Does Not Exist")

    return dataloader

