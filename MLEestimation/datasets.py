from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import torch




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

