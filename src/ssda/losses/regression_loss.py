from torch import nn

def mse_loss_regression(predictions, targets):
    return nn.functional.mse_loss(predictions, targets)