from dataclasses import dataclass

@dataclass
class SSVAETrainerConfig:
    name:str = "SSVAETrainer"
    learning_rate: float = 1e-3
    number_of_epochs: int = 10
    save_model_epochs: int = 5
    debug:bool = False

    experiment_class: str = "mnist"
    device:str = "cuda:0"
