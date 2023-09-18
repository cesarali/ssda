import torch
import unittest

# configs
from ssda.configs.vae_config import VAEConfig

#load
from ssda.data.dataloader_utils import load_dataloader
from ssda.models.models_utils import load_encoder

class TestEncoder(unittest.TestCase):

    def test_encoder(self):
        z_dim = 23
        batch_size = 128
        expected_size = torch.Size([batch_size,z_dim])

        config = VAEConfig()
        config.z_dim = z_dim
        config.dataloader.batch_size = batch_size

        dataloader = load_dataloader(config)
        databatch = next(dataloader.train().__iter__())
        data,_ = databatch

        encoder = load_encoder(config)
        z,mu,logvar = encoder(data)

        self.assertEqual(z.shape,expected_size)
        self.assertEqual(mu.shape,expected_size)
        self.assertEqual(logvar.shape,expected_size)

        print(z.shape)
        print(mu.shape)
        print(logvar.shape)



if __name__=="__main__":
    unittest.main()