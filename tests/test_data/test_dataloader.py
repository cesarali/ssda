import torch
import unittest
from pprint import pprint
from dataclasses import asdict
from ssda.data.dataloader_utils import load_dataloader
from ssda.data.dataloaders_config import NISTLoaderConfig
from ssda.configs.vae_config import get_config_from_file
from ssda.configs.vae_config import VAEConfig

class TestNISTDataloader(unittest.TestCase):

    def setUp(self):
        self.batch_size = 23
        self.config = VAEConfig(experiment_name='ssda',
                                experiment_type='mnist',
                                experiment_indentifier="vae_unittest",
                                delete=True)

        self.config.dataloader = NISTLoaderConfig(batch_size=128)

    def test_dataloader(self):
        dataloader = load_dataloader(self.config)
        databatch = next(dataloader.train().__iter__())
        data,_ = databatch
        print(data.shape)


class TestSemisupervisedDataloader(unittest.TestCase):

    def test_mnist_dataloader(self):
        from ssda.configs.ssvae_config import SSVAEConfig
        from ssda.data.dataloaders_config import SemisupervisedLoaderConfig

        self.batch_size = 23

        self.config = SSVAEConfig(experiment_name='ssvae',
                                  experiment_type='mnist',
                                  experiment_indentifier="ssvae_unittest")

        self.config.dataloader = SemisupervisedLoaderConfig(batch_size=128)

        dataloader = load_dataloader(self.config)
        databatch = next(dataloader.train(type="unlabeled").__iter__())
        data,_ = databatch
        print(data.shape)

if __name__=="__main__":
    unittest.main()
