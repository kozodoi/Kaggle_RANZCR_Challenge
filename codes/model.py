####### MODEL PREP

import timm
import torch
import torch.nn as nn


def get_model(CFG, 
              device):
    
    '''
    Instantiate the model
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'


    if CFG['weights'] != 'public':

        # convolutional part
        model = timm.create_model(model_name = CFG['backbone'], 
                                  pretrained = False if CFG['weights'] == 'empty' else True,
                                  in_chans   = CFG['channels'])

        # classifier part                            
        if 'efficient' in CFG['backbone']:
            model.classifier = nn.Linear(model.classifier.in_features, CFG['num_classes'])
        elif 'vit' in CFG['backbone']:
            model.head = nn.Linear(model.head.in_features, CFG['num_classes'])
        else:
            model.fc = nn.Linear(model.fc.in_features, CFG['num_classes'])
        

    elif CFG['weights'] == 'public':
        
        # custom model class
        class CustomModel(nn.Module):

            def __init__(self, model_name='resnet200d', out_dim=11, pretrained=False):
                super().__init__()
                self.model             = timm.create_model(model_name, pretrained=False)
                n_features             = self.model.fc.in_features
                self.model.global_pool = nn.Identity()
                self.model.fc          = nn.Identity()
                self.pooling           = nn.AdaptiveAvgPool2d(1)
                self.fc                = nn.Linear(n_features, out_dim)

            def forward(self, x):
                bs              = x.size(0)
                features        = self.model(x)
                pooled_features = self.pooling(features).view(bs, -1)
                output          = self.fc(pooled_features)
                return output
    
        # initialize
        model = CustomModel(CFG['backbone'], CFG['num_classes'], False)

    # wrapper for TPU
    if CFG['device'] == 'TPU':
        model = xmp.MpModelWrapper(model)

    return model