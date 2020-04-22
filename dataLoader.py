#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torchvision as tv
import torch

def load_dataset(path): 
#    torch.manual_seed(0)
    dataset = tv.datasets.ImageFolder(
            root=path,
            transform=tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
                                                      tv.transforms.ToTensor(),
                                                      tv.transforms.Normalize((0.5,), (0.5, ))
                                                      ])
    )

    return dataset





def load_dataloader(path): 
#    torch.manual_seed(0)
    dataset = tv.datasets.ImageFolder(
            root=path,
            transform=tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
                                                      tv.transforms.ToTensor(),
                                                      tv.transforms.Normalize((0.5,), (0.5, ))
                                                      ])
    )
#    train_dataset, test_dataset = load_dataset(path)
#    torch.manual_seed(torch.initial_seed())
    
    loader = torch.utils.data.DataLoader(
            dataset,
            #batch_size=50,
            num_workers=0,
            shuffle=True
            )
    

    return dataset, loader





