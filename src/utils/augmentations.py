from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose, ToGray, InvertImg, HorizontalFlip, RandomCrop, IAAPerspective
)

from albumentations.core.transforms_interface import DualTransform

import imgaug as ia
from imgaug import augmenters as iaa

import numpy as np
import cv2
import torch


class Crop(DualTransform):
    """Crops region from image.
    Args:
        window (tuple (int, int)): 
        central (bool): 
        p (float [0, 1]): 
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    def __init__(self, window, central=False, p=1.0):
        super(Crop, self).__init__(p)
        self.window = np.array(window)
        self.central = central

    def augment_image(self, im, anchors=None):
        if self.central:
            point  = np.array([
                (im.shape[0] - self.window[0]) // 2,
                (im.shape[1] - self.window[1]) // 2
            ])
        elif anchors is not None:
            point = anchors[np.random.randint(len(anchors))] - self.window // 2
            point = np.clip(point, 0, np.array(im.shape[:2]) - self.window)
        else:
            point = np.array([
                np.random.randint(0, max(1, im.shape[0] - self.window[0] + 1)),
                np.random.randint(0, max(1, im.shape[1] - self.window[1] + 1))
            ])

        return im[
            point[0]: point[0] + self.window[0], 
            point[1]: point[1] + self.window[1]
        ]

    def apply(self, img, anchors=None, **params):
        return self.augment_image(img, anchors)


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(2, 3), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(2, 3), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(2, 3), k=3))
    im = np.array(im)[:, :, :, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(2, 3), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(2, 3), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(2, 3), k=3))
    return np.concatenate(mirrs, axis=0)


def get_crops(image, side):
    im_side = image.shape[-1]
    assert im_side >= side
    margin = (im_side - side) // 2
    images = [
        image[..., :side, :side],
        image[..., -side:, -side:],
        image[..., :side, -side:],
        image[..., -side:, :side],
        image[..., 
            margin: side + margin, 
            margin: side + margin
        ]
    ]
    return torch.cat(images)


class SimplexNoise:
    def __init__(self):
        self.simplex_noise = iaa.Sometimes(
            .9,
            iaa.SimplexNoiseAlpha(
                first=iaa.Sometimes(.7, iaa.MedianBlur(k=iaa.Choice([0, 5]))),
                second=iaa.Sometimes(.7, iaa.Multiply(iaa.Choice([0.5, 1.5]), per_channel=False)),
                upscale_method="linear"
            )
        )
    def __call__(self, image):
        return self.simplex_noise.augment_image(image)


class Augmentation:
    def __init__(self, side=None, strength=1., simplex_noise=False):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        self.simplex_noise = SimplexNoise()
        self.strength = strength
        self.side = side
        self.augs = self.get_augmentations()
        self.crop = Crop(window=(side, side), central=strength is None)

    def __call__(self, data):
        if data['mask'] is not None:
            data['image'] = np.dstack([data['image'], data['mask']])
        data['image'] = self.crop.apply(data['image'], data['anchors'])
        data.update({
            'image': data['image'][..., :3],
            'mask': data['image'][..., 3:],
        })
#         if self.simplex_noise:
#             data['image'] = self.simplex_noise(data['image'])
        if self.augs is not None:
            data = self.augs(**data)
        return data

    def get_photometric(self):
        coeff = int(3 * self.strength)
        k = max(1, coeff if coeff % 2 else coeff - 1)

        return Compose([
            OneOf([
                CLAHE(clip_limit=2, p=.4),
                IAASharpen(p=.5),
                IAAEmboss(p=.5),
            ], p=0.8),
            OneOf([
                IAAAdditiveGaussianNoise(p=.3),
                GaussNoise(p=.7),
            ], p=.8),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=k, p=.3),
                Blur(blur_limit=k, p=.5),
            ], p=.8),
            OneOf([
                RandomContrast(),
                RandomBrightness(),
            ], p=.7)
        ])

    def get_geoometric(self):
        geometric = [
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-.35, .35), rotate_limit=45, p=.95),
            IAAPerspective(scale=(.05, .2), keep_size=True, p=.8),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.3),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.75)
        ]
        return Compose(geometric)

    def get_augmentations(self):
        if self.strength is None:
            return None

        transformations = [
            Compose([
                RandomRotate90(),
                Flip(),
                Transpose(),
            ], p=1.),
            Compose([
                self.get_photometric(),
                self.get_geoometric(),
            ], p=1)
        ]

#         if self.side is not None:
#             transformations.append(
#                 RandomCrop(self.side, self.side)
#             )

        return Compose(transformations)
