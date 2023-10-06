import os
import torch
import pickle
import unittest
from ssda import data_path
from ssda.data.porous_dataloaders import PorousDataLoaderDict

class TestPorousDict(unittest.TestCase):

    def test_porous_dict(self):
        from ssda.data.porous_dataloaders_config import PorousDataLoaderConfig

        config = PorousDataLoaderConfig()
        config.batch_size = 2

        dataloader = PorousDataLoaderDict(config, "patients")
        databatch = next(dataloader.train().__iter__())
        print(databatch)

        dataloader = PorousDataLoaderDict(config, "simulations")
        databatch = next(dataloader.train().__iter__())
        print(databatch)

@unittest.skip
class TestPorousLoader(unittest.TestCase):

    def setUp(self):
        from ssda.configs.ssvae_config import SSVAEConfig
        from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig

        self.batch_size = 23
        self.config = SSVAEConfig(experiment_indentifier="ssvae_porous_unittest",delete=True)

        self.config.dataloader = SemisupervisedLoaderPorousConfig(batch_size=14)

    def test_porous_dict(self):
        from ssda.data.dataloader_utils import load_dataloader
        device = torch.device("cpu")

        dataloader = load_dataloader(self.config)
        databatch = next(dataloader.train(data_type="unlabel").__iter__())

        print(f"X {databatch[0].shape} Y {databatch[1].shape}")


if __name__=="__main__":
    unittest.main()
