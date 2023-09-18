from dataclasses import dataclass

@dataclass
class ClassifierConfig:
    name:str = "Classifier"
    classifier_hidden_size:int = 400