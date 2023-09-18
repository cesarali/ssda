
import unittest
from ssda.configs.vae_config import VAEConfig
from ssda.trainers.vae_trainer import VAETrainer

class TestMITrainer(unittest.TestCase):

    read_config = VAEConfig

    def test_trainer(self):
        config = VAEConfig(experiment_indentifier="vae_trainer_unittest")
        config.trainer.device = "cuda:0"
        config.trainer.number_of_epochs = 1

        vae_trainer = VAETrainer(config)
        vae_trainer.train()

if __name__=="__main__":
    unittest.main()
