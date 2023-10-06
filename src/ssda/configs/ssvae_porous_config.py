from pprint import pprint
from dataclasses import dataclass, asdict
from ssda.configs.ssvae_config import SSVAEConfig
from ssda.data.porous_dataloaders_config import SemisupervisedLoaderPorousConfig

@dataclass
class SSPorousVAEConfig(SSVAEConfig):
    experiment_type:str = 'porous'
    experiment_name:str = 'ssvae'
    dataloader: SemisupervisedLoaderPorousConfig = SemisupervisedLoaderPorousConfig()

    def __post_init__(self):
        super().__post_init__()
        self.classifier_loss_type = "mse"

if __name__=="__main__":
    config = SSPorousVAEConfig()
    pprint(asdict(config))