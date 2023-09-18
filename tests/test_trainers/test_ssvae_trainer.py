import unittest
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.trainers.ssvae_trainer import SSVAETrainer

class TestMITrainer(unittest.TestCase):

    read_config = SSVAEConfig

    def test_trainer(self):
        config = SSVAEConfig(experiment_indentifier="vae_trainer_unittest")
        config.trainer.device = "cuda:0"
        config.trainer.number_of_epochs = 1

        vae_trainer = SSVAETrainer(config)
        vae_trainer.initialize()

if __name__=="__main__":
    unittest.main()
