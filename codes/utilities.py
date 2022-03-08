####### UTILITIES

import os
import numpy as np
import random
import torch
from sklearn.metrics import roc_auc_score


# competition metric
def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


# device-aware printing
def smart_print(expression, CFG):
    if CFG['device'] != 'TPU':
        print(expression)
    else:
        xm.master_print(expression)


# device-aware model save
def smart_save(weights, path, CFG):
    if CFG['device'] != 'TPU':
        torch.save(weights, path)    
    else:
        xm.save(weights, path) 


# randomness
def seed_everything(seed, CFG):
    assert isinstance(seed, int), 'seed has to be an integer'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    smart_print('- setting random seed to {}...'.format(seed), CFG)