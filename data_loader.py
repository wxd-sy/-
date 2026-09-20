import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    dataset: str = "mnist",
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if dataset == "mnist":
        ds_cls = datasets.MNIST
    elif dataset == "fashion_mnist":
        ds_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_ds = ds_cls(data_dir, train=True, download=True, transform=transform)
    test_ds = ds_cls(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
