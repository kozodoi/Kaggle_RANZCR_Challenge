####### LOSS PREP

import torch.nn as nn


class FocalCosineLoss(nn.Module):
    
    def __init__(self, alpha = 1, gamma = 2, xent = 0.1, reduction = "mean"):
        super(FocalCosineLoss, self).__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.xent      = xent
        self.reduction = reduction
        self.y         = torch.Tensor([1]).to(device)
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, target, self.y, reduction = self.reduction)
        cent_loss   = nn.BCEWithLogitsLoss()(input, target)
        pt          = torch.exp(-cent_loss)
        focal_loss  = self.alpha * (1-pt)**self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        
        return cosine_loss + self.xent * focal_loss


def get_losses(CFG, device, epoch = None):
    
    '''
    Get training and validation loss functions
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # define training loss
    if CFG['loss_fn'] == 'BCE':
        train_criterion = nn.BCEWithLogitsLoss().to(device)
    elif CFG['loss_fn'] == 'FC':
        train_criterion = FocalCosineLoss()
    
    # define valid loss
    valid_criterion = nn.BCEWithLogitsLoss().to(device)

    return train_criterion, valid_criterion