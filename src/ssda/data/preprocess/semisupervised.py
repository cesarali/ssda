import torch
from torch.utils.data import Subset,DataLoader


# Function to create semi-supervised dataloader
def create_semi_supervised_dataloader(mnist_dataset, labeled_proportion, batch_size):
    num_samples = len(mnist_dataset)
    num_labeled = int(num_samples * labeled_proportion)
    num_unlabeled = num_samples - num_labeled

    # Shuffle the dataset to mix labeled and unlabeled samples
    indices = torch.arange(num_samples)

    # Create a subset of labeled data
    labeled_subset = Subset(mnist_dataset, indices[:num_labeled])

    # Create a subset of unlabeled data
    unlabeled_subset = Subset(mnist_dataset, indices[num_labeled:])

    # Create data loaders for labeled and unlabeled data
    labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader