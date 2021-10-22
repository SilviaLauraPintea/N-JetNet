import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import random
import numpy as np

class dataCIFAR:
    def __init__(self, dataset, batch, train=True, val=True, workers=4):
        if val==True: assert(train==True)

        if dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            self.num_classes = 10
        else:
            dataloader = datasets.CIFAR100
            self.num_classes = 100

        # Data transformations  
        if train:   
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
 
        dataset = dataloader(root='./data', train=train, download=True, 
                            transform=transform)

        
        # Define some random indices for training and val splits
        if train:
            indices = list(range(len(dataset)))
            np.random.RandomState(10).shuffle(indices)

            if val:
                indices = indices[len(indices) // 2:]
                self.loader = data.DataLoader(dataset, batch_size=batch,
                                sampler=data.sampler.SubsetRandomSampler(indices),
                                num_workers=workers)
            else:
                indices = indices[:len(indices) // 2]
                self.loader = data.DataLoader(dataset, batch_size=batch,
                                sampler=data.sampler.SubsetRandomSampler(indices),
                                num_workers=workers)
        else:
            self.loader = data.DataLoader(dataset, batch_size=batch, 
                                shuffle=train, 
                                num_workers=workers)



