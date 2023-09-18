from typing import Union

from ssda.configs.vae_config import VAEConfig
from ssda.data.dataloaders_config import NISTLoaderConfig

from ssda.data.dataloaders import NISTLoader

def load_dataloader(config:Union[VAEConfig,NISTLoaderConfig]):
    config_:NISTLoaderConfig

    if isinstance(config,VAEConfig):
        config_ = config.dataloader
    elif isinstance(config, NISTLoaderConfig):
        config_ = config
    else:
        raise Exception("Config Does Not Exist")

    if config_.name == "NISTLoader":
        dataloader = NISTLoader(config_)
    else:
        raise Exception("Dataloader Does Not Exist")

    return dataloader

