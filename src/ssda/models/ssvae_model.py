import os
import sys
from dataclasses import dataclass, asdict
from typing import Union

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

# loss
from ssda.losses.contrastive_loss import vae_loss
from ssda.losses.classifier_loss import classifier_loss_cross_entropy

EPSILON = 1e-12

@dataclass
class SSVAELoss:
    reconstruction_loss:Union[float,torch.Tensor]
    kl_loss:Union[float,torch.Tensor]
    classifier_loss:Union[float,torch.Tensor]

class SSVAE(nn.Module):

    config: SSVAEConfig

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
            self.create_new_from_config(self.config,device)
        elif read:
            self.load_results_from_directory(experiment_name=experiment_name,
                                             experiment_type=experiment_type,
                                             experiment_indentifier=experiment_indentifier,
                                             checkpoint=checkpoint,
                                             device=device)

    def loss(self,forward_pass,databatch,data_type="label"):
        if data_type == "label":
            x = databatch[0]
            label =  databatch[1]

            recon_x, mu, logvar, logits = forward_pass

            vae_loss_ = self.vae_loss(recon_x, x , mu, logvar)
            classifier_loss_ = self.classifier_loss(logits, label)

            loss_ = self.config.vae_loss_lambda*vae_loss_ + self.config.classifier_loss_lambda*classifier_loss_
        else:
            x = databatch[0]
            recon_x, mu, logvar = forward_pass
            vae_loss_ = self.vae_loss(recon_x, x, mu, logvar)
            loss_ = vae_loss_
        return loss_

    def forward(self, databath, data_type="label", inference=True):
        if data_type == "label":
            image = databath[0]
            z, mu, logvar = self.encoder(image)
            logits = self.classifier(z)
            forward_ = self.decoder(z), mu, logvar, logits
        elif data_type == "unlabel":
            image = databath[0]
            z, mu, logvar = self.encoder(image)
            forward_ = self.decoder(z),mu,logvar

        if inference:
            return forward_
        else:
            loss_ = self.loss(forward_,databath,data_type=data_type)
            return forward_,loss_

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

        self.define_loss()

    def load_results_from_directory(self,
                                    experiment_name='ssvae',
                                    experiment_type='mnist',
                                    experiment_indentifier="test",
                                    checkpoint=None,
                                    device=torch.device("cpu")):

        self.config, self.encoder,self.decoder, self.classifier,self.dataloader = load_ssvae_experiments_configuration(experiment_name,
                                                                                                                       experiment_type,
                                                                                                                       experiment_indentifier,
                                                                                                                       checkpoint)
        self.encoder.to(device)
        self.decoder.to(device)
        self.classifier.to(device)


    def define_loss(self):
        #set other stuff
        if self.config.vae_loss_type == "vae_loss":
            self.vae_loss = vae_loss
        else:
            raise Exception("Loss Not Implemented")

        #set other stuff
        if self.config.classifier_loss_type == "classifier_loss":
            self.classifier_loss = classifier_loss_cross_entropy
        else:
            raise Exception("Loss Not Implemented")
