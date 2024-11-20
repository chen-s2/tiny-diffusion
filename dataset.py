import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

def create_dataloader(data_path, image_size, batch_size, dataset_name):
    transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]

    transform = transforms.Compose(transforms_list)

    dataset = ImageFolder(data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    print(f"Dataset contains {len(dataset):,} images ({data_path})")

    loader.dataset_name = dataset_name
    return loader