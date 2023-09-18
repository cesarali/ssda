import unittest
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.trainers.ssvae_trainer import SSVAETrainer

from ssda.models.ssvae_model import SSVAE

class TestMITrainer(unittest.TestCase):

    read_config = SSVAEConfig

    @unittest.skip
    def test_trainer(self):
        config = SSVAEConfig(experiment_indentifier="ssvae_trainer_classifier")
        config.trainer.device = "cuda:0"

        config.trainer.number_of_epochs = 10
        config.dataloader.labeled_proportion = 0.5

        config.vae_loss_lambda = 1.
        config.classifier_loss_lambda = 1.

        vae_trainer = SSVAETrainer(config)
        vae_trainer.train()

    def test_ssvae_read_results(self):
        ssvae = SSVAE()
        ssvae.load_results_from_directory(experiment_name='ssvae',
                                          experiment_type='mnist',
                                          experiment_indentifier="ssvae_trainer_classifier",
                                          checkpoint=None)

if __name__=="__main__":
    unittest.main()
