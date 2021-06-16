####### DATASET

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import pandas as pd

from utilities import smart_print
from augmentations import get_augs

from torch.utils.data import Dataset
class ImageData(Dataset):
    
    # initialization
    def __init__(self, 
                 df, 
                 path, 
                 transform = None, 
                 labeled   = False,
                 indexed   = False):
        self.df        = df
        self.path      = path
        self.transform = transform
        self.labeled   = labeled
        self.indexed   = indexed
        
    # length
    def __len__(self):
        return len(self.df)
    
    # get item  
    def __getitem__(self, idx):
        
        # import
        path  = os.path.join(self.path, self.df.iloc[idx]['StudyInstanceUID'] + '.jpg')
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
            
        # crop gray areas
        if CFG['crop_borders']:
            mask  = image > 0
            image = image[np.ix_(mask.any(1), mask.any(0))]
            
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
        # augmentations
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        # convert RGB to grayscale
        if CFG['channels'] == 1:
            image = image[0, :, :]
            image = np.expand_dims(image, 0)
            
        # output
        if self.labeled:
            label = torch.tensor(self.df.iloc[idx][CFG['targets']]).float()
            if self.indexed:
                idx = torch.tensor(idx)
                return idx, image, label
            else: 
                return image, label
        else:
            return image



####### DATA PREP

def get_data(df, fold, CFG):

    # load splits
    df_train = df.loc[df.fold != fold].reset_index(drop = True)
    df_valid = df.loc[df.fold == fold].reset_index(drop = True)     
    smart_print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)

    # psueudo-labeled data
    if CFG['data_pl']:
        df_train = pd.concat([df_train, df_pl], axis = 0).reset_index(drop = True)
        smart_print('- appending 2019 pseudo-labeled data to train...', CFG)
        smart_print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)

    # subset for debug mode
    if CFG['debug']:
        df_train = df_train.sample(CFG['batch_size'] * 8, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 8, random_state = CFG['seed']).reset_index(drop = True)

    return df_train, df_valid




####### LOADERS PREP

def get_loaders(df_train, df_valid, CFG, epoch = None):

    ##### EPOCH-BASED PARAMS

    # image size
    if (CFG['step_size']) and (epoch is not None):
        image_size = CFG['step_size'][epoch]
    else:
        image_size = CFG['image_size']

    # augmentation probability
    if (CFG['step_p_aug']) and (epoch is not None):
        p_aug = CFG['step_p_aug'][epoch]
    else:
        p_aug = CFG['p_aug']
        

    ##### DATASETS
        
    # augmentations
    train_augs, valid_augs = get_augs(CFG, image_size, p_aug)

    # datasets
    train_dataset = ImageData(df        = df_train, 
                              path      = CFG['data_path'] + 'train/',
                              transform = train_augs,
                              labeled   = True,
                              indexed   = False)
    valid_dataset = ImageData(df        = df_valid, 
                              path      = CFG['data_path'] + 'train/',
                              transform = valid_augs,
                              labeled   = True,
                              indexed   = True)
    
    
    ##### DATA SAMPLERS
    
    # GPU samplers
    if CFG['device'] != 'TPU':
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        
    # TPU samplers
    elif CFG['device'] == 'TPU':
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas = xm.xrt_world_size(),
                                           rank         = xm.get_ordinal(),
                                           shuffle        = True)
        valid_sampler = DistributedSampler(valid_dataset,
                                           num_replicas = xm.xrt_world_size(),
                                           rank         = xm.get_ordinal(),
                                           shuffle        = False)
        
    ##### DATA LOADERS
       
    # data loaders
    train_loader = DataLoader(dataset     = train_dataset, 
                              batch_size  = CFG['batch_size'], 
                              sampler     = train_sampler,
                              num_workers = CFG['cpu_workers'],
                              pin_memory  = True)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['batch_size'], 
                              sampler     = valid_sampler, 
                              num_workers = CFG['cpu_workers'],
                              pin_memory  = True)
    
    # feedback
    smart_print('- image size: {}x{}, p(augment): {}'.format(image_size, image_size, p_aug), CFG)
    if epoch is None:
        smart_print('-' * 55, CFG)
    
    return train_loader, valid_loader