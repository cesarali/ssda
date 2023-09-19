import os
import torch
import pickle
import unittest
from ssda import data_path
from ssda.data.porous_dataloaders import PorousDataLoaderDict

@unittest.skip
class TestPorousDict(unittest.TestCase):

    def test_porous_dict(self):
        simulations_dir = os.path.join(data_path, "raw", "porous", "simulation")
        simulations_path = os.path.join(simulations_dir, "simulations_.cp")

        """
        with open("./data/patient_.cp","rb") as file:
            clients_data = pickle.load(file)
            #clients_data["y_diff"] = torch.Tensor(clients_data["y_diff"])
            #client_data_loader = PorousDataLoader(clients_data,type="client")
        """

        with open(simulations_path, "rb") as file:
            simulations_data = pickle.load(file)
            simulations_data["nds"] = torch.Tensor(simulations_data["nds"])

        simulations_data_loader = PorousDataLoaderDict(simulations_data, type="simulations")
        databatch = next(simulations_data_loader.train().__iter__())
        # print(databatch.keys())
        print(databatch['PoDmD'].shape)
        print(torch.isnan(databatch['nds']).any())


class TestPorousLoader(unittest.TestCase):

    def setUp(self):
        from ssda.configs.ssvae_config import SSVAEConfig
        from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig

        self.batch_size = 23
        self.config = SSVAEConfig(experiment_name='ssvae',
                                  experiment_type='porous',
                                  experiment_indentifier="ssvae_porous_unittest",
                                  delete=True)

        self.config.dataloader = SemisupervisedLoaderPorousConfig(batch_size=14)

    def test_porous_dict(self):
        from ssda.data.dataloader_utils import load_dataloader
        device = torch.device("cpu")

        dataloader = load_dataloader(self.config)
        databatch = next(dataloader.train(data_type="unlabel").__iter__())

        print(f"X {databatch[0].shape} Y {databatch[1].shape}")


if __name__=="__main__":
    unittest.main()
