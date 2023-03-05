from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import torch
from IPython import embed
import torchvision
from torchvision.transforms import ToTensor


def snelson_dataloader(path_to_data:"str", batch_size:int):
  """
  Args:
      path_to_data (string): Path to folder containing the snelson input and outputs
  """
  x= torch.from_numpy(np.genfromtxt(path_to_data +"snelson_train_inputs.txt"))
  y= torch.from_numpy(np.genfromtxt(path_to_data +"snelson_train_outputs.txt"))
  x, y= x.to(torch.float32), y.to(torch.float32)
  x= x.view(x.size(0),-1)
  y = y.view(y.size(0), -1)
  dataset= TensorDataset(x, y)
  dataloader= DataLoader(dataset, batch_size=batch_size)
  return dataloader



def mnist_dataloader(train: bool, batch_size:int):
  dataset = torchvision.datasets.MNIST(root="", train=train, transform=ToTensor(), download=True)
  x, y= dataset.data, dataset.targets
  x= x.view(x.size(0), -1)
  x, y= x.to(torch.float32), y.to(torch.long)
  dataset= TensorDataset(x, y)
  dataloader= DataLoader(dataset, batch_size= batch_size)
  return dataloader

def dataloader_classification(name: "str", train: bool, batch_size:int):
  if name== "CIFAR10":
    dataset = torchvision.datasets.CIFAR10(root="", train=train, transform=ToTensor(), download=True)
  elif name=="CIFAR100":
    dataset = torchvision.datasets.CIFAR100(root="", train=train, transform=ToTensor(), download=True)
  elif name == "FMNIST":
    dataset = torchvision.datasets.FashionMNIST(root="", train=train, transform=ToTensor(), download=True)
  elif name == "MNIST":
    dataset = torchvision.datasets.MNIST(root="", train=train, transform=ToTensor(), download=True)
  x, y= dataset.data, dataset.targets
  x= x.view(x.size(0), -1)
  x, y= x.to(torch.float32), y.to(torch.long)
  dataset= TensorDataset(x, y)
  dataloader= DataLoader(dataset, batch_size= batch_size)
  return dataloader


