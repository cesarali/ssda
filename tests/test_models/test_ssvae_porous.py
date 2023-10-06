import torch
import unittest

from ssda.models.ssvae_model import SSVAE
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.configs.ssvae_porous_config import SSPorousVAEConfig

from ssda.data.dataloader_utils import load_dataloader

class TestSSPorousVAE(unittest.TestCase):

    read_config = SSVAEConfig

    @unittest.skip
    def test_ssvae(self):
        z_dim = 23
        batch_size = 128
        config = SSPorousVAEConfig(experiment_indentifier="ssvae_porous_unittest")
        config.z_dim = z_dim
        config.dataloader.batch_size = batch_size
        config.trainer.device = "cpu"

        device = torch.device(config.trainer.device)

        ssvae = SSVAE()
        ssvae.create_new_from_config(config,device)
        dataloader = load_dataloader(config)

        #=====================================
        # INFERENCE
        #=====================================
        """
        databatch = next(dataloader.train(data_type="label").__iter__())

        reconstruction, mu, logvar,logits = ssvae(databatch,data_type="label")
        print(f"Reconstruction {reconstruction.shape} Mu {mu.shape} Logits {logits.shape}")

        reconstruction, mu, logvar = ssvae(databatch,data_type="unlabel")
        print(f"Reconstruction {reconstruction.shape}")
        """
        #=====================================
        # TRAINING
        #=====================================
        databatch = next(dataloader.train(data_type="label").__iter__())
        (reconstruction, mu, logvar,logits),loss = ssvae(databatch,data_type="label",inference=False)
        print(f"Loss {reconstruction.shape} Mu {mu.shape} Logits {logits.shape}")

        databatch = next(dataloader.train(data_type="unlabel").__iter__())
        (reconstruction, mu, logvar),loss = ssvae(databatch,data_type="unlabel",inference=False)
        print(f"Reconstruction {reconstruction.shape}")

if __name__=="__main__":
    unittest.main()

