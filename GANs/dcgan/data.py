import torch
import config
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

def create_dataset():
    dataset = datasets.CIFAR10(root="./", download=True,
                            transform=transforms.Compose([
                                transforms.Resize(config.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batchSize,
                                            shuffle=True)
    
    return dataset, dataloader



