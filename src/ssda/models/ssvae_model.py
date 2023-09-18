import os
import sys
from dataclasses import dataclass, asdict

# torch stuff
import torch
from torch import nn

# load models
from ssda.data.dataloader_utils import load_dataloader
from ssda.models.models_utils import load_encoder,load_decoder,load_classifier
from ssda.trainers.trainers_utils import load_ssvae_experiments_configuration

# configs
from ssda.configs.ssvae_config import SSVAEConfig

# models
from ssda.models.decoder import Decoder
from ssda.models.encoder import Encoder
from ssda.data.dataloaders import NISTLoader

EPSILON = 1e-12

class SSVAE(nn.Module):

    def __init__(self,
                 config:SSVAEConfig=None,
                 experiment_name='ssvae',
                 experiment_type='mnist',
                 experiment_indentifier="test",
                 checkpoint=None,
                 device=torch.device("cpu"),
                 read=False):
        super(SSVAE,self).__init__()

        self.config = config
        if self.config is not None:
            self.create_new_from_config()
        elif read:
            self.load_results_from_directory(experiment_name=experiment_name,
                                             experiment_type=experiment_type,
                                             experiment_indentifier=experiment_indentifier,
                                             checkpoint=checkpoint,
                                             device=device)

    def forward(self,databath,type="label"):
        if type == "label":
            image = databath[0]
            z, mu, logvar = self.encoder(image)
            logits = self.classifier(z)
            return self.decoder(z),mu,logvar,logits
        elif type == "unlabel":
            image = databath[0]
            z, mu, logvar = self.encoder(image)
            return self.decoder(z),mu,logvar

    def to(self,device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)

    def generate(self, number_of_samples=64):
        # Generating samples from the trained VAE
        self.eval()
        with torch.no_grad():
            z_sample = torch.randn(number_of_samples, self.config.z_dim)
            sample = self.decoder(z_sample).cpu()
        return sample

    #===============================
    # files and stuff
    #===============================
    def create_new_from_config(self, config:SSVAEConfig, device=torch.device("cpu")):
        self.config = config
        self.config.initialize_new_experiment()

        self.dataloader = load_dataloader(self.config)

        self.encoder = load_encoder(self.config)
        self.encoder.to(device)

        self.decoder = load_decoder(self.config)
        self.decoder.to(device)

        self.classifier = load_classifier(self.config)
        self.classifier.to(device)

    def load_results_from_directory(self,
                                    experiment_name='ssda',
                                    experiment_type='mnist',
                                    experiment_indentifier="test",
                                    checkpoint=None,
                                    device=torch.device("cpu")):

        self.encoder,self.decoder, self.dataloader = load_ssvae_experiments_configuration(experiment_name,
                                                                                          experiment_type,
                                                                                          experiment_indentifier,
                                                                                          checkpoint)
        self.encoder.to(device)
        self.decoder.to(device)


if __name__=="__main__":
    from ssda.utils.plots import plot_sample

    """
    ssva = SSVAE()
    ssva.load_results_from_directory(experiment_name='ssvae',
                                    experiment_type='mnist',
                                    experiment_indentifier="vae_train_example",
                                    checkpoint=None)
    sample_ = ssva.generate(number_of_samples=64)
    plot_sample(sample_)
    """

