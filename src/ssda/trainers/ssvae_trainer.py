import torch
from torch.optim import Adam

import numpy as np
from pprint import pprint
from dataclasses import asdict

from ssda.data.dataloaders import NISTLoader

from ssda.models.ssvae_model import SSVAE
from ssda.configs.ssvae_config import SSVAEConfig


from ssda.models.encoder_config import EncoderConfig

class SSVAETrainer:

    name_="ssvae_trainer"

    def __init__(self,
                 config: SSVAEConfig,
                 dataloader:NISTLoader=None,
                 vae:SSVAE=None):

        #set parameter values
        self.config = config
        self.learning_rate = config.trainer.learning_rate
        self.number_of_epochs = config.trainer.number_of_epochs
        self.device = torch.device(config.trainer.device)

        #define models
        self.ssvae = SSVAE()
        self.ssvae.create_new_from_config(self.config, self.device)
        self.dataloader = self.ssvae.dataloader


    def parameters_info(self):
        print("# ==================================================")
        print("# START OF BACKWARD SSVAE TRAINING ")
        print("# ==================================================")
        print("# SSVAE parameters ************************************")
        pprint(asdict(self.ssvae.config))
        print("# Paths Parameters **********************************")
        pprint(asdict(self.dataloader.config))
        print("# Trainer Parameters")
        pprint(asdict(self.config))
        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def preprocess_data(self,data_batch):
        return (data_batch[0].to(self.device), data_batch[1].to(self.device))

    def train_step(self,data_batch,number_of_training_step,data_type="label"):
        data_batch = self.preprocess_data(data_batch)
        forward_pass,loss_ = self.ssvae(data_batch,data_type=data_type,inference=False)

        self.optimizer.zero_grad()
        loss_.backward()
        self.optimizer.step()

        self.writer.add_scalar('{0}/training loss'.format(data_type), loss_, number_of_training_step)
        return loss_

    def test_step(self,data_batch,data_type="label"):
        with torch.no_grad():
            data_batch = self.preprocess_data(data_batch)
            forward_pass,loss_ = self.ssvae(data_batch,data_type=data_type,inference=False)
            return loss_

    def initialize(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)

        self.optimizer = Adam(self.ssvae.parameters(), lr=self.learning_rate)

        # TEST LABEL DATA
        data_batch = next(self.dataloader.train(data_type="label").__iter__())
        data_batch = self.preprocess_data(data_batch)
        forward_,initial_loss_label = self.ssvae(data_batch,data_type="label",inference=False)

        assert torch.isnan(initial_loss_label).any() == False
        assert torch.isinf(initial_loss_label).any() == False

        # TEST UNLABEL DATA
        data_batch = next(self.dataloader.train(data_type="unlabel").__iter__())
        data_batch = self.preprocess_data(data_batch)
        forward_,initial_loss_unlabel = self.ssvae(data_batch,data_type="unlabel",inference=False)

        assert torch.isnan(initial_loss_unlabel).any() == False
        assert torch.isinf(initial_loss_unlabel).any() == False

        self.save_results(self.ssvae,
                          initial_loss_label,
                          average_train_loss_unlabel=None,
                          average_train_loss_label=None,
                          average_test_loss_label=None,
                          LOSS=None,
                          epoch=0,
                          checkpoint=True)

        return initial_loss_label

    def train(self):
        initial_loss_label = self.initialize()
        best_loss = initial_loss_label

        number_of_training_step_unlabel = 0
        number_of_training_step_label = 0
        number_of_test_step = 0
        for epoch in range(self.number_of_epochs):
            print(f"EPOCH {epoch} out of {self.number_of_epochs}")
            LOSS = []
            #UNLABEL TRAINING----------------------------
            train_loss_unlabel = []
            for data_batch in self.dataloader.train(data_type="unlabel"):
                loss = self.train_step(data_batch,number_of_training_step_unlabel,data_type="unlabel")
                train_loss_unlabel.append(loss.item())
                LOSS.append(loss.item())
                number_of_training_step_unlabel += 1
                if number_of_training_step_unlabel % 100 == 0:
                    print("UNLABELED/number_of_training_step: {}, Loss: {}".format(number_of_training_step_unlabel, loss.item()))
            average_train_loss_unlabel = np.asarray(train_loss_unlabel).mean()

            #LABELED TRAINING----------------------------
            train_loss_label = []
            for data_batch in self.dataloader.train(data_type="label"):
                loss = self.train_step(data_batch,number_of_training_step_label,data_type="label")
                train_loss_label.append(loss.item())
                LOSS.append(loss.item())
                number_of_training_step_label += 1
                if number_of_training_step_label % 100 == 0:
                    print("LABEL/number_of_training_step: {}, Loss: {}".format(number_of_training_step_label, loss.item()))
            average_train_loss_label = np.asarray(train_loss_label).mean()

            # TEST LABEL TRAINING-------------------------
            test_loss_label = []
            for data_batch in self.dataloader.test(data_type="label"):
                loss = self.test_step(data_batch)
                test_loss_label.append(loss.item())
                number_of_test_step+=1
            average_test_loss_label = np.asarray(test_loss_label).mean()

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if average_test_loss_label < best_loss:
                self.save_results(self.ssvae,
                                  initial_loss_label,
                                  average_train_loss_unlabel=average_train_loss_unlabel,
                                  average_train_loss_label=average_train_loss_label,
                                  average_test_loss_label=average_test_loss_label,
                                  LOSS=LOSS,
                                  epoch=epoch,
                                  checkpoint=False)

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                self.save_results(self.ssvae,
                                  initial_loss_label,
                                  average_train_loss_unlabel=average_train_loss_unlabel,
                                  average_train_loss_label=average_train_loss_label,
                                  average_test_loss_label=average_test_loss_label,
                                  LOSS=LOSS,
                                  epoch=epoch,
                                  checkpoint=True)

        self.writer.close()

    def save_results(self,
                     ssvae,
                     initial_loss_label,
                     average_train_loss_unlabel,
                     average_train_loss_label,
                     average_test_loss_label,
                     LOSS,
                     epoch=0,
                     checkpoint=False):
        if checkpoint:
            RESULTS = {
                "model":ssvae,
                "initial_loss_label":initial_loss_label,
                "average_train_loss_unlabel":average_train_loss_unlabel,
                "average_train_loss_label":average_train_loss_label,
                "average_test_loss_label":average_test_loss_label,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path_checkpoint.format(epoch))
        else:
            RESULTS = {
                "model":ssvae,
                "initial_loss_label":initial_loss_label,
                "average_train_loss_unlabel":average_train_loss_unlabel,
                "average_train_loss_label":average_train_loss_label,
                "average_test_loss_label":average_test_loss_label,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path)



