import torch
from torch import nn
from ssda.configs.vae_config import VAEConfig
from ssda.configs.ssvae_config import SSVAEConfig

class Classifier(nn.Module):
    def __init__(self,config:SSVAEConfig):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(config.z_dim, config.classifier.classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(config.classifier.classifier_hidden_size, config.dataloader.number_of_labels)  # 10 classes in MNIST
        )

    def forward(self, z):
        logits = self.classifier(z)
        return logits

