import os
import torch

class UnsqueezeTensorTransform:

    def __init__(self,axis=0):
        self.axis = axis
    def __call__(self, tensor:torch.Tensor):
        return tensor.unsqueeze(self.axis)