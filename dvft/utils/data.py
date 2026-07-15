import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_imagenet_loader(data_dir='/data', batch_size=32):
    """ImageNet-1K 数据加载"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=0.2)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    return train_loader, val_loader, 1000


def get_cifar100_loader(data_dir='./data', batch_size=32):
    """CIFAR-100 数据加载"""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=0.2)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    val_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, 100