from torch import nn

def classifier_loss_cross_entropy(logits,labels):
    return nn.functional.cross_entropy(logits, labels)