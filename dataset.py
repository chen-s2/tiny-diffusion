import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)

class LineDrawingTransform:
    def __init__(self, threshold1=50, threshold2=150):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, img):
        # Convert the PIL Image to a NumPy array
        img_np = np.array(img)

        # Convert to grayscale
        gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, self.threshold1, self.threshold2)

        # Invert the edges to get black on white line drawing
        line_drawing = cv2.bitwise_not(edges)

        # Convert the NumPy array back to a PIL Image
        return Image.fromarray(line_drawing)

def create_dataloader(data_path, image_size, batch_size, dataset_name):
    transforms_list = [
        transforms.Resize((image_size, image_size)),
        LineDrawingTransform(threshold1=50, threshold2=150),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
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

def create_dataloader_orig(data_path, image_size, batch_size, dataset_name, gray):
    transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
    if gray:
        transforms_list.append(transforms.Grayscale())

    transform = transforms.Compose(transforms_list)

    dataset = ImageFolder(data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        # num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"Dataset contains {len(dataset):,} images ({data_path})")

    loader.dataset_name = dataset_name
    return loader

def create_dataloader_cifar(image_size, batch_size, train_or_test='train'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        transforms.Grayscale()
    ])

    train_dataset = CIFAR10(root='./data', train=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, transform=transform)

    if train_or_test == 'train':
        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("loaded cifar data with batches #:", len(dataloader), "and images:", len(dataloader) * batch_size)
    dataloader.dataset_name = 'cifar10'
    return dataloader
