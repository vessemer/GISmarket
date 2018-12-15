from skimage.morphology import label
import cv2
import os
import numpy as np
from tqdm import tqdm


def multi_rle_encode(labels):
    # labels = label(img[:, :, 0])
    drops = np.where(np.bincount(labels.flatten()) < 1190.47333333)[0]
    for drop in tqdm(drops):
        labels[labels == drop] = 0
    return [rle_encode(labels==k) for k in tqdm(np.unique(labels[labels>0]))]


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = True
    return img.reshape(shape)  # Needed to align to RLE direction


def masks_as_image(in_mask_list, shape=(768, 768)):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(shape, dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def read_masks(mask_rles, shape=(768, 768)):
    masks = np.zeros((shape[0], shape[1]))
    for idx, rle in enumerate(mask_rles):
        if isinstance(rle, str):
            masks[rle_decode(rle, shape)] = idx + 1
    return masks


def init_masks(root, df, shape=(768, 768)):
    try:
        os.mkdir(root)
    except FileExistsError:
        pass

    keys = np.unique(df.ImageId.values)
    for key in tqdm(keys):
        eps = df.query('ImageId==@key').EncodedPixels
        mask = masks_as_image(eps, shape)
        key = key.replace('.jpg', '.png')
        cv2.imwrite(os.path.join(root, key), np.squeeze(mask) * 255)
