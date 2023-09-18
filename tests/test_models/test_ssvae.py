import torch
import unittest

from ssda.models.ssvae_model import SSVAE
from ssda.configs.ssvae_config import SSVAEConfig

from ssda.data.dataloader_utils import load_dataloader

class TestSSVAE(unittest.TestCase):

    read_config = SSVAEConfig

    def test_ssvae(self):
        z_dim = 23
        batch_size = 128
        expected_size = torch.Size([batch_size,z_dim])

        config = SSVAEConfig(experiment_indentifier="ssvae_unittest")
        config.z_dim = z_dim
        config.dataloader.batch_size = batch_size
        config.trainer.device = "cpu"

        device = torch.device(config.trainer.device)

        ssvae = SSVAE()
        ssvae.create_new_from_config(config,device)

        dataloader = load_dataloader(config)
        databatch = next(dataloader.train().__iter__())

        reconstruction, mu, logvar,logits = ssvae(databatch,type="label")
        print(f"Reconstruction {reconstruction.shape} Mu {mu.shape} Logits {logits.shape}")

        #reconstruction, mu, logvar = ssvae(databatch,type="unlabel")
        #print(reconstruction.shape)

if __name__=="__main__":
    unittest.main()
