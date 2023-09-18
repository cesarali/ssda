from dataclasses import dataclass

@dataclass
class SSVAETrainerConfig:
    name:str = "SSVAETrainer"
    learning_rate: float = 1e-3
    number_of_epochs: int = 10
    save_model_epochs: int = 5

    vae_loss_type:str = "vae_loss"
    classifier_loss_type:str = "classifier_loss"

    experiment_class: str = "mnist"
    device:str = "cuda:0"
