####### TRAINING

import timm
from timm.utils import *

from utilities import *

import torch
from tqdm import tqdm

import numpy as np


def train_epoch(loader, model, optimizer, scheduler, criterion, epoch, CFG, device):
       
    # switch regime
    model.train()

    # running loss
    trn_loss = AverageMeter()

    # update scheduler on epoch
    if not CFG['update_on_batch']:
        scheduler.step() 
        if epoch == CFG['warmup']:
            scheduler.step() 

    # loop through batches
    for batch_idx, (inputs, labels) in (tqdm(enumerate(loader), total = len(loader)) if CFG['device'] != 'TPU' \
                                        else enumerate(loader)):

        # extract inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # label smoothing
        if CFG['smooth']:
            for image_idx, label in enumerate(labels):
                labels[image_idx] = torch.where(label == 1.0, 1.0 - CFG['smooth'], CFG['smooth'])

        # update scheduler on batch
        if CFG['update_on_batch']:
            scheduler.step(epoch + 1 + batch_idx / len(loader))

        # passes and weight updates
        with torch.set_grad_enabled(True):
            
            # forward pass 
            with amp_autocast():
                preds = model(inputs)
                loss  = criterion(preds, labels)
                loss  = loss / CFG['accum_iter']
                
            # backward pass
            if CFG['use_amp'] and CFG['device'] == 'GPU':
                scaler.scale(loss).backward()   
            else:
                loss.backward() 

            # update weights
            if ((batch_idx + 1) % CFG['accum_iter'] == 0) or ((batch_idx + 1) == len(loader)):
                if CFG['device'] == 'TPU':
                    xm.optimizer_step(optimizer, barrier = True)
                else:
                    if CFG['use_amp'] and CFG['device'] == 'GPU':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()

        # update loss
        trn_loss.update(loss.item() * CFG['accum_iter'], inputs.size(0))

        # clear memory
        del inputs, labels, preds, loss
        gc.collect()

    return trn_loss.sum



####### INFERENCE

def valid_epoch(loader, model, criterion, CFG, device):

    # switch regime
    model.eval()

    # running loss
    val_loss = AverageMeter()

    # placeholders
    IDS   = []
    PROBS = []
       
    # loop through batches
    with torch.no_grad():
        for batch_idx, (ids, inputs, labels) in (tqdm(enumerate(loader), total = len(loader)) if CFG['device'] != 'TPU' \
                                        else enumerate(loader)):

            # extract inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)

            # preds placeholders
            logits = torch.zeros((inputs.shape[0], CFG['num_classes']), device = device)
            probs  = torch.zeros((inputs.shape[0], CFG['num_classes']), device = device)

            # compute predictions
            for tta_idx in range(CFG['num_tta']): 
                preds   = model(get_tta_flips(inputs, tta_idx))
                logits += preds / CFG['num_tta']
                probs  += preds.sigmoid() / CFG['num_tta']

            # compute loss
            loss = criterion(logits, labels)
            val_loss.update(loss.item(), inputs.size(0))

            # store predictions
            IDS.append(ids.detach().cpu())
            PROBS.append(probs.detach().cpu())

            # clear memory
            del ids, inputs, labels, probs, logits, preds, loss
            gc.collect()

    # transform predictions
    IDS   = torch.cat(IDS).numpy()
    PROBS = torch.cat(PROBS).numpy()

    return val_loss.sum, IDS, PROBS