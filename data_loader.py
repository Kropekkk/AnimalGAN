from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_transform(size):
  data_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

  return data_transform

def create_dataloader(
    train_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
):

  train_data = datasets.ImageFolder(train_dir, transform=transform)

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
  )

  return train_dataloader

def load_data(
    train_dir: str,
    batch_size: int,):
  
  data_transform = create_transform(64)
  dataloader = create_dataloader(train_dir, data_transform, batch_size)

  return dataloader
