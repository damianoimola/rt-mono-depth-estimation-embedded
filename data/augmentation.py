import torch
from torchvision.transforms import transforms
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np



class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean




class ElasticTransform(object):
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        random_state = np.random.RandomState(None)
        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)



def augment_data():
    # horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
    # random_crop = transforms.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.33))
    # color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # gaussian_noise = AddGaussianNoise(mean=0.0, std=0.1)
    # random_rotation = transforms.RandomRotation(degrees=10)
    # random_scale = transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))
    # elastic_transform = ElasticTransform(alpha=1.0, sigma=0.08)

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # AddGaussianNoise(mean=0.0, std=0.1),
        # transforms.RandomRotation(degrees=10),
        # transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        # ElasticTransform(alpha=1.0, sigma=0.08),
        # transforms.ToTensor(),
    ])

