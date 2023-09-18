import torch
import unittest

# configs
from ssda.configs.vae_config import VAEConfig

# load
from ssda.data.dataloader_utils import load_dataloader
from ssda.models.models_utils import load_decoder
from ssda.models.models_utils import load_encoder


class TestDecoder(unittest.TestCase):

    def test_decoder(self):
        z_dim = 23
        batch_size = 128
        expected_encoder_size = torch.Size([batch_size, z_dim])


        config = VAEConfig()
        config.z_dim = z_dim
        config.dataloader.batch_size = batch_size
        expected_decoder_size = torch.Size([batch_size, config.dataloader.input_dim])

        dataloader = load_dataloader(config)
        databatch = next(dataloader.train().__iter__())
        data, _ = databatch

        encoder = load_encoder(config)
        decoder = load_decoder(config)

        z, mu, logvar = encoder(data)
        decoded_image = decoder(z)

        self.assertEqual(z.shape, expected_encoder_size)
        self.assertEqual(mu.shape, expected_encoder_size)
        self.assertEqual(logvar.shape, expected_encoder_size)
        self.assertEqual(decoded_image.shape,expected_decoder_size)


if __name__ == "__main__":
    unittest.main()