import os
import cv2
import numpy as np
import pandas as pd

import sklearn.model_selection

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose




class UrbanDataset(Dataset):
    def __init__(self, file_names, basepath, augmentations=None):
        self.transform = augmentations
        self.basepath = basepath
        self.file_names = file_names

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        data = {"image": image}
        if self.transform is not None:
            augmented = self.transform(data)
            image = augmented["image"]

        return self.postprocess(idx, image, mask)

    def postprocess(self, idx, image, mask):
        return { 
            'image': image.astype(np.uint8),
            'pid': self.file_names[idx],
            'mask': mask
        }

    def load_image(self, idx):
        path = self.basepath + self.file_names[idx]
        path = path.replace("GTI", "RGB")
        image = cv2.imread(path, -1)
        return image

    def load_mask(self, idx):
        path = self.basepath + self.file_names[idx]
        mask = cv2.imread(path, -1)
        return mask

    def __len__(self):
        return len(self.file_names)