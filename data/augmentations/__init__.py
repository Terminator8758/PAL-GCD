from torchvision import transforms
import torch
from PIL import Image, ImageFilter
import random


def get_transform(transform_type='imagenet', image_size=32, args=None, mean=None, std=None):

    if transform_type == 'imagenet':

        if mean==None or std==None:
            mean = (0.485, 0.456, 0.406)  # original dino-pretrained model
            std = (0.229, 0.224, 0.225)

        interpolation = args.interpolation
        crop_pct = args.crop_pct  # default: 0.875 (default image_size=224)

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:
        raise NotImplementedError

    return (train_transform, test_transform)



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
