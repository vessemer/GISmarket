import torch

import numpy as np
from tqdm import tqdm
import numpy_indexed as npi

from ..configs import config
from . import visualisation
from . import dataset as ds

import matplotlib.pyplot as plt


def get_model(model, checkpoint=None, map_location=None, devices=None):
    model.cuda()

    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location)#.module.state_dict()
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd}
        print('Overlapped keys: {}'.format(len(sd.keys())))
        msd.update(sd)
        model.load_state_dict(msd)

    if devices is not None:
        model = torch.nn.DataParallel(model, device_ids=devices)

    return model


def freeze(model, unfreeze=False):
    children = list(model.children())
    if hasattr(model, 'children') and len(children):
        for child in children:
            freeze(child, unfreeze)
    elif hasattr(model, 'parameters'):
        for param in model.parameters():
            param.requires_grad = unfreeze


def unfreeze_bn(model):
    if isinstance(model, torch.nn.BatchNorm2d):
        for param in model.parameters():
            param.requires_grad = True

    children = list(model.children())
    if len(children):
        for child in children:
            unfreeze_bn(child)
    return None


class GlobalPooling(torch.nn.Module):
    def __init__(self, pooling_type='avg', dropout=None):
        super(GlobalPooling, self).__init__()
        if pooling_type == 'avg':
            self.pooling = torch.mean
        if pooling_type == 'max':
            self.pooling = torch.max
        else:
            assert 'Provided incorrect `pooling_type`'
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        x = self.pooling(x.view(x.size(0), x.size(1), -1), dim=2)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def set_io_dims(model, in_channels=4, out_channels=28, 
                pooling_type='avg', dropout=None):
#     weights = model.conv1.state_dict()['weight']
#     pholder = torch.zeros((weights.size(0), in_channels, *weights.shape[2:]))
#     pholder[:, :3] = weights

    # Create input conv layer
    conv1 = torch.nn.Conv2d(
        in_channels, 
        model.conv1.out_channels, 
        kernel_size=model.conv1.kernel_size, 
        stride=model.conv1.stride, 
        padding=model.conv1.padding, 
        bias=model.conv1.bias is not None
    )

#     sd = conv1.state_dict()
#     sd['weight'] = pholder
#     conv1.load_state_dict(sd)

    # If input channels == 1, then keep summed weights
    if in_channels == 1:
        nstd = dict()
        for key, param in model.conv1.state_dict().items():
            nstd[key] = param.sum(1).unsqueeze(1)
        conv1.load_state_dict(nstd)
    model.conv1 = conv1

    # Use global pooling instead 
    model.avgpool = GlobalPooling(pooling_type=pooling_type, dropout=dropout)

    # Create and reassign output linear layer
    model.fc = torch.nn.Linear(
        model.fc.in_features, 
        out_channels, 
        model.fc.bias is not None
    )
    return model


def set_io_dims_inception(model, in_channels=4):
    conv1 = model.Conv2d_1a_3x3.conv
    # Create input conv layer
    conv1 = torch.nn.Conv2d(
        in_channels, 
        conv1.out_channels, 
        kernel_size=conv1.kernel_size, 
        stride=conv1.stride, 
        padding=conv1.padding, 
        bias=conv1.bias is not None
    )

    # If input channels == 1, then keep summed weights
    if in_channels == 1:
        nstd = dict()
        for key, param in model.conv1.state_dict().items():
            nstd[key] = param.sum(1).unsqueeze(1)
        conv1.load_state_dict(nstd)

    model.Conv2d_1a_3x3.conv = conv1
    return model


def set_io_dims_densnet(model, in_channels=4, out_channels=28, 
                pooling_type='avg', dropout=None):
    # densnet
    conv1 = model.features.conv0

    # Create input conv layer
    conv1 = torch.nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None
    )

    model.features.conv0 = conv1

    # Use global pooling instead
    # model.avgpool = GlobalPooling(pooling_type=pooling_type, dropout=dropout)

    fc = model.classifier

    # Create and reassign output linear layer
    model.classifier = torch.nn.Linear(
        fc.in_features, 
        out_channels, 
        fc.bias is not None
    )
    return model


class Learner:
    def __init__(self, model, loss, opt=None, metrics=[], gclip=1.):
        self.gclip = gclip
        self.model = model
        self.opt = opt
        self.loss = loss
        self.metrics = metrics
        
        self.infer_ids = np.arange(config.PARAMS['NB_INFERS'])

        if self.opt is not None:
            for group in self.opt.param_groups:
                group.setdefault('initial_lr', group['lr'])

    def make_step(self, data, training=False):
        self.opt.zero_grad()
        image = torch.autograd.Variable(data['image']).cuda(non_blocking=True)
        mask = torch.autograd.Variable(data['mask']).cuda(non_blocking=True)

        prediction = self.model(image)
        loss = self.loss(prediction, mask)
        prediction = torch.sigmoid(prediction)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gclip)
            self.opt.step()

        image = image.data.cpu().numpy()
        return {
            'loss': loss.data.cpu().numpy(), 
        }

    def train_on_epoch(self, datagen, hard_negative_miner=None, lr_scheduler=None):
        torch.cuda.empty_cache()
        self.model.train()
        meters = list()

        for data in tqdm(datagen):
            meters.append(self.make_step(data, training=True))
            if lr_scheduler is not None:
                if hasattr(lr_scheduler, 'batch_step'):
                    lr_scheduler.batch_step(logs=meters[-1])

            if hard_negative_miner is not None:
                hard_negative_miner.update_cache(meters[-1], data)
                if hard_negative_miner.need_iter():
                    self.make_step(hard_negative_miner.get_cache(), training=True)
                    hard_negative_miner.invalidate_cache()

        return meters
    
    def validate(self, datagen):
        torch.cuda.empty_cache()
        self.model.eval()
        meters = list()

        with torch.no_grad():
            for data in tqdm(datagen):
                meters.append(self.make_step(data, training=False))

        return meters

    def save(self, path):
        state_dict = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)
        print('Saved in {}:'.format(path))

    def infer_on_data(self, data, verbose=True):
        if self.model.training:
            self.model.eval()
        if len(data['image'].shape) == 3:
            data['image'] = data['image'].unsqueeze(0)

        with torch.no_grad():
            image = torch.autograd.Variable(data['image']).cuda()
            pred = self.model.forward(image)
            pred = torch.sigmoid(pred).data.cpu().numpy()
            if verbose:
                image = np.rollaxis(data['image'][0].numpy(), 0, 3)
                image = (image * ds.STD + ds.MEAN)
                mask = np.zeros_like(image)
                mask[..., :2] = np.rollaxis(data['mask'].numpy(), 0, 3) * 255
                _, ax = plt.subplots(ncols=3, figsize=(15, 5))
                ax[0].imshow(image)
                ax[1].imshow(mask)
                pred_ = np.zeros_like(image)
                pred_[..., :2] = np.rollaxis(pred[0], 0, 3)
                ax[2].imshow(pred_)
                plt.show()

        return pred
