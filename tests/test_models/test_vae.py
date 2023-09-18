

import torch
import unittest

from ssda.configs.vae_config import VAEConfig
from ssda.models.vae_model import VAE

from ssda.data.dataloader_utils import load_dataloader


class TestVAE(unittest.TestCase):

    read_config = VAEConfig

    def test_vae(self):
        z_dim = 23
        batch_size = 128
        expected_size = torch.Size([batch_size,z_dim])

        config = VAEConfig(experiment_indentifier="vae_unittest")
        config.z_dim = z_dim
        config.dataloader.batch_size = batch_size
        config.trainer.device = "cpu"

        device = torch.device(config.trainer.device)

        vae = VAE()
        vae.create_new_from_config(config,device)

        dataloader = load_dataloader(config)
        databatch = next(dataloader.train().__iter__())
        data,_ = databatch

        reconstruction, mu, logvar = vae(data)
        print(reconstruction.shape)



if __name__=="__main__":
    unittest.main()

