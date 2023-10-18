import unittest
from ssda.configs.vae_config import VAEConfig
from ssda.trainers.vae_trainer import VAETrainer

class TestVAETrainer(unittest.TestCase):

    read_config = VAEConfig

    def test_trainer(self):
        config = VAEConfig(experiment_indentifier="vae_trainer_unittest",
                           experiment_type="vae")
        config.trainer.device = "cuda:0"
        config.trainer.number_of_epochs = 1
        config.trainer.debug = True

        vae_trainer = VAETrainer(config)
        vae_trainer.train()

    @unittest.skip
    def test_vae_porous(self):
        from ssda.data.porous_dataloaders_config import PorousDataLoaderConfig
        z_dim = 2
        batch_size = 32
        encoder_hidden_size = 100
        decoder_hidden_size = 100
        type = "patients"

        config = VAEConfig(experiment_indentifier=f"encoder_{encoder_hidden_size}_decoder_{decoder_hidden_size}_type_{type}",
                           experiment_type="porous",
                           z_dim=z_dim)
        config.dataloader = PorousDataLoaderConfig(batch_size=batch_size,
                                                   type=type)

        config.encoder.stochastic = False
        config.encoder.encoder_hidden_size = encoder_hidden_size
        config.decoder.decoder_hidden_size = decoder_hidden_size

        config.trainer.device = "cuda:0"
        config.trainer.debug = False
        config.trainer.number_of_epochs = 30
        config.trainer.save_model_epochs = int(.25*config.trainer.number_of_epochs)
        config.trainer.learning_rate = 1e-3

        vae_trainer = VAETrainer(config)
        vae_trainer.train()

    @unittest.skip
    def test_load_vae(self):
        from ssda.models.vae_model import VAE
        encoder_hidden_size = 100
        decoder_hidden_size = 100

        type = "patients"
        patients_vae = VAE()
        patients_vae.load_results_from_directory(experiment_name="vae",
                                                 experiment_type="porous",
                                                 experiment_indentifier=f"encoder_{encoder_hidden_size}_decoder_{decoder_hidden_size}_type_{type}")

        type = "simulations"
        simulations_vae = VAE()
        simulations_vae.load_results_from_directory(experiment_name="vae",
                                                 experiment_type="porous",
                                                 experiment_indentifier=f"encoder_{encoder_hidden_size}_decoder_{decoder_hidden_size}_type_{type}")

        for databatch in simulations_vae.dataloader.train():
            rec_0,mu,logavr = patients_vae.encoder(databatch["images"].float())
            rec_1,mu,logavr = simulations_vae.encoder(databatch["images"].float())
            print(rec_0.shape)


        for databatch in patients_vae.dataloader.train():
            rec,mu,logavr = patients_vae.encoder(databatch["images"].float())
            print(rec)



if __name__=="__main__":
    unittest.main()
