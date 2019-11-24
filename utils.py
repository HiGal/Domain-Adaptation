from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_source_dataloader(path, batch_size, transform):
    dataset_source = datasets.SVHN(root=path, download=True, transform=transform)
    dataset_source_val = datasets.SVHN(root=path, split='test', download=True, transform=transform)
    source_dataloader = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=4)
    source_dataloader_val = DataLoader(dataset=dataset_source_val, batch_size=batch_size, shuffle=True, num_workers=4)
    return source_dataloader, source_dataloader_val


def get_target_dataloader(path, batch_size, transform):
    dataset_target = datasets.MNIST(root=path, download=True, transform=transform)
    dataset_target_val = datasets.MNIST(root=path, train=False, download=True, transform=transform)
    target_dataloader = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=4)
    target_dataloader_val = DataLoader(dataset=dataset_target_val, batch_size=batch_size, shuffle=True, num_workers=4)
    return target_dataloader, target_dataloader_val
