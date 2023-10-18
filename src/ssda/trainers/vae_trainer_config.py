from dataclasses import dataclass

@dataclass
class VAETrainerConfig:
    name:str = "VAETrainer"
    learning_rate: float = 1e-3
    number_of_epochs: int = 10
    save_model_epochs: int = None
    debug:bool = False

    loss_type:str = "vae_loss"
    experiment_class: str = "mnist"
    device:str = "cuda:0"

    def __post_init__(self):
        self.save_model_epochs = int(0.25*self.number_of_epochs)
