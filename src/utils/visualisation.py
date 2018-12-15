import os

import numpy as np
from skimage.io import imread


def load_image(basepath, image_id):
    images = np.zeros(shape=(4,512,512))
    basepath = os.path.join(basepath, image_id)
    images[0,:,:] = imread(basepath + "_green.png")
    images[1,:,:] = imread(basepath + "_red.png")
    images[2,:,:] = imread(basepath + "_blue.png")
    images[3,:,:] = imread(basepath + "_yellow.png")
    return images

def make_image_row(image, subax, title):
    subax[0].imshow(image[0], cmap="Blues")
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax

def make_title(labels, label_names, file_id):
    file_targets = labels.loc[labels.Id==file_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title


class TargetGroupIterator:
    
    def __init__(self, labels, reverse_train_labels, target_names, batch_size, basepath):
        self.target_names = target_names
        self.labels = labels
        self.target_list = [reverse_train_labels[key] for key in target_names]
        self.batch_shape = (batch_size, 4, 512, 512)
        self.basepath = basepath
    
    def find_matching_data_entries(self):
        self.labels["check_col"] = self.labels.Target.apply(
            lambda l: self.check_subset(l)
        )
        self.images_identifier = self.labels[self.labels.check_col==1].Id.values
        self.labels.drop("check_col", axis=1, inplace=True)
    
    def check_subset(self, targets):
        return np.where(set(targets).issubset(set(self.target_list)), 1, 0)
    
    def get_loader(self):
        filenames = []
        idx = 0
        images = np.zeros(self.batch_shape)
        for image_id in self.images_identifier:
            images[idx,:,:,:] = load_image(self.basepath, image_id)
            filenames.append(image_id)
            idx += 1
            if idx == self.batch_shape[0]:
                yield filenames, images
                filenames = []
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield filenames, images