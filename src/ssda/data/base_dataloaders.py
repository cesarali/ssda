import torch
from abc import ABC
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split

class BasicDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DictDataSet(Dataset):
    """
    # Define your data dictionary
    data_dict = {'input': torch.randn(2, 10), 'target': torch.randn(2, 5)}

    # Create your dataset
    my_dataset = DictDataSet(data_dict)

    # Create a DataLoader from your dataset
    batch_size = 2
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    """
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data_dict[self.keys[0]])

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.keys}

class BaseDataLoader(ABC):
    name_="base_data_loader"
    def __init__(self,**kwargs):
        super(BaseDataLoader,self).__init__()

    def define_dataset_and_dataloaders(self,X,training_proportion=0.8,batch_size=32):
        self.batch_size = batch_size
        if isinstance(X,torch.Tensor):
            dataset = TensorDataset(X)
        elif isinstance(X,dict):
            dataset = DictDataSet(X)

        self.total_data_size = len(dataset)
        self.training_data_size = int(training_proportion * self.total_data_size)
        self.test_data_size = self.total_data_size - self.training_data_size

        training_dataset, test_dataset = random_split(dataset, [self.training_data_size, self.test_data_size])
        self._train_iter = DataLoader(training_dataset, batch_size=batch_size)
        self._test_iter = DataLoader(test_dataset, batch_size=batch_size)

    def train(self):
        return self._train_iter

    def test(self):
        return self._test_iter