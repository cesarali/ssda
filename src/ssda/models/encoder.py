import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from ssda.configs.vae_config import VAEConfig

# Define the VAE model
class Encoder(nn.Module):
    def __init__(self,config:VAEConfig):
        super(Encoder, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(self.config.dataloader.input_dim, self.config.encoder.encoder_hidden_size)
        self.fc21 = nn.Linear(self.config.encoder.encoder_hidden_size, self.config.z_dim)  # Mean
        self.fc22 = nn.Linear(self.config.encoder.encoder_hidden_size, self.config.z_dim)  # Variance


    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
