import os
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from glob import glob
import sklearn.model_selection


MEAN = [0.31611712, 0.30189482, 0.20617615]
STD = [0.06082994, 0.06330029, 0.0722641]

img_transform = Compose([
    ToTensor(),
    Normalize(mean=MEAN, std=STD)
])


class AR_Dataset(Dataset):
    def __init__(self, paths, augmentations=None):
        self.transform = augmentations
        self.paths = paths.copy()
        self.keys = list(self.paths.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        meta = self.paths[self.keys[idx]]
        image = cv2.imread(meta['image'])
        mask = cv2.imread(meta['mask'])[..., :2]

        anchors = None
        if 'anchors' in meta:
            anchors = meta['anchors']

        data = {"image": image, "mask": mask, "anchors": anchors}
        if self.transform is not None:
            augmented = self.transform(data)
            image, mask = augmented["image"], augmented["mask"]

        mask = (mask > 120).astype(np.float32)
            
        return { 
            'image': img_transform(image),
            'mask': torch.from_numpy(np.rollaxis(mask, 2, 0)),
        }


def get_folds(nbs, seed):
    paths = glob('../data/TRAINING/*')
    paths = np.array([os.path.basename(path) for path in paths])
    folds = sklearn.model_selection.KFold(n_splits=nbs, random_state=seed, shuffle=True)
    folds = list(folds.split(paths))
    return [(paths[fold[0]], paths[fold[1]]) for fold in folds]


def build_paths(names, prefix=''):
    paths = list()
    for name in names:
        paths.extend(glob('../data/TRAINING/{}/{}_*_*{}_image.png'.format(name, name, prefix)))
    paths = { os.path.basename(v) : { 'image': v, 'mask': v.replace('image', 'borders') } for v in paths }
    return paths


def get_datasets(folds, fold, prefix='', augmentations=None):
    train, valid = folds[fold]
    train, valid = build_paths(train, prefix), build_paths(valid, prefix)
    train = AR_Dataset(train, augmentations=augmentations)
    valid = AR_Dataset(valid, augmentations=None)
    return train, valid


def get_datagens(dataset_train, dataset_val, batch_size=32):
    train_datagen = torch.utils.data.DataLoader(dataset_train, 
                                                pin_memory=True,
                                                shuffle=True,
                                                batch_size=batch_size,
                                                num_workers=4)
    val_datagen = torch.utils.data.DataLoader(dataset_val,
                                              pin_memory=True,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              num_workers=4)
    return train_datagen, val_datagen


# class _Dataset(Dataset):
#     def __init__(self, paths, augmentations=None):
#         self.transform = augmentations
#         self.paths = paths.copy()
#         self.keys = list(self.paths.keys())

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         meta = self.paths[self.keys[idx]]
#         image = cv2.imread(meta['image'])
#         mask = cv2.imread(meta['mask'], 0)

#         if mask is None:
#             mask = np.zeros(image.shape[:-1], dtype=np.uint8)
#         mask = np.expand_dims(mask, -1)

#         anchors = None
#         if 'anchors' in meta:
#             anchors = meta['anchors']

#         data = {"image": image, "mask": mask, "anchors": anchors}
#         if self.transform is not None:
#             augmented = self.transform(data)
#             image, mask = augmented["image"], augmented["mask"]

#         mask = (mask > 120).astype(np.float32)
            
#         return { 
#             'image': img_transform(image),
#             'mask': torch.from_numpy(np.rollaxis(mask, 2, 0)),
#         }
