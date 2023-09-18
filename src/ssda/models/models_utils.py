from typing import Union
from ssda.configs.vae_config import VAEConfig
from ssda.models.encoder import Encoder
from ssda.models.decoder import Decoder

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