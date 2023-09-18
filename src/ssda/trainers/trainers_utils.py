import torch
from ssda.configs.vae_config import VAEConfig
from ssda.configs.vae_config import get_config_from_file

def load_experiments_results(experiment_name, experiment_type, experiment_indentifier, checkpoint:int = None):
    config: VAEConfig
    config = get_config_from_file(experiment_name, experiment_type, experiment_indentifier)
    if checkpoint is None:
        results = torch.load(config.experiment_files.best_model_path)
    else:
        results = torch.load(config.experiment_files.best_model_path_checkpoint.format(checkpoint))
    return config, results

def load_experiments_configuration(experiment_name, experiment_type, experiment_indentifier, checkpoint:int = None):
    from ssda.data.dataloader_utils import load_dataloader

    config: VAEConfig
    config, results = load_experiments_results(experiment_name, experiment_type, experiment_indentifier, checkpoint)
    vae_model = results["ssda"]
    dataloader = load_dataloader(config)

    return vae_model.encoder,vae_model.decoder,dataloader