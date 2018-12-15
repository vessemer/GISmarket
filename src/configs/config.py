import os 
import numpy as np
import pandas as pd


DATA_ROOT = '../data'

PATHS = {
#     'TRAIN': 'train_full_size',
    'TRAIN': 'train',
    'EXT': 'external_data',
    'TEST': 'test_full_size',
    'CSV': 'csv',
    'XML': 'xmls'
}

for k, v in PATHS.items():
    PATHS[k] = os.path.join(DATA_ROOT, v)

PATHS['DATA'] = DATA_ROOT

PARAMS = {
    'PATHS': PATHS,
    'SEED': 42,
    'NB_INFERS': 3,
    'NB_FOLDS': 10,
    'SUPPORT_CLASS_AMOUNT': 700,
    'SUPPORT_POWER': .7,
    'BATCH_SIZE': 16, #28,
    'VALID_BATCH_SIZE': 28,
    'DROPOUT': .5,
    'NB_EPOCHS': 50,
    'EPOCHS_PER_SAVE': 10,
    'NB_FREEZED_EPOCHS': 0,
    
    'SIMPLEX_NOISE': True,
    'SIDE': 448,

    'LR': 5e-4,
    'MIN_LR': 1e-4,
    'EXP_GAMMA': .95,

    'CUDA_DEVICES': [0, 1],
    'LR_POLICE': [[0, 5e-4], [8, 1e-5], [12, 5e-6], [45, 1e-6]],
    'PLOT_KEYS': [
        'loss'
    ],

    'SHRINKED_FULL_SIZE': 1024,
    'SHRINKED_SIDE': 512,
    'USE_EXTERNAL_DATA': True,
}
