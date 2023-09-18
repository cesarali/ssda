from typing import Union
from ssda.configs.vae_config import VAEConfig
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.models.encoder import Encoder
from ssda.models.decoder import Decoder
from ssda.models.classifier import Classifier


def load_encoder(config:VAEConfig):
    if config.encoder.name == "Encoder":
        encoder = Encoder(config)
    else:
        raise Exception("No Classifier")
    return encoder

def load_decoder(config:VAEConfig):
    if config.encoder.name == "Encoder":
        decoder = Decoder(config)
    else:
        raise Exception("No Classifier")
    return decoder

def load_classifier(config:SSVAEConfig):
    if config.classifier.name == "Classifier":
        decoder = Classifier(config)
    else:
        raise Exception("No Classifier")
    return decoder