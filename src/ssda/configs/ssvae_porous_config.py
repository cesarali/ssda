from pprint import pprint
from dataclasses import dataclass, asdict
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig

class SSPorousVAEConfig(SSVAEConfig):

    def __post_init__(self):
        super().__post_init__()
        self.experiment_type = 'porous'

        self.dataloader = SemisupervisedLoaderPorousConfig(batch_size=self.dataloader.batch_size)
        self.classifier_loss_type = "mse"

if __name__=="__main__":
    config = SSPorousVAEConfig()
    pprint(asdict(config))