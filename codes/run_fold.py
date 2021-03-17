####### WRAPPER FUNCTION

def run_fold(fold, df_trn, df_val, CFG, model, device):

    ##### PREPARATIONS
    
    # reset seed
    seed_everything(CFG['seed'] + fold, CFG)
    
    # update device
    if CFG['device'] == 'TPU':
        device = xm.xla_device()
    model = model.to(device)
        
    # get optimizer
    optimizer, scheduler = get_optimizer(CFG, model)
    
    # get loaders
    trn_loader, val_loader = get_loaders(df_trn, df_val, CFG)
        
    # placeholders
    trn_losses  = []
    val_losses  = []
    val_metrics = []
    lrs         = []

    
    ##### TRAINING AND INFERENCE

    for epoch in range(CFG['num_epochs'] + CFG['fine_tune']):
                
        ### PREPARATIONS

        # timer
        epoch_start = time.time()

        # update data loaders if needed
        if (CFG['step_size']) or (CFG['step_p_aug']):
            trn_loader, val_loader = get_loaders(df_trn, df_val, CFG, epoch)   

        # update freezing for fine-tuning if needed
        if (CFG['fine_tune']) and (epoch == CFG['num_epochs']):
            smart_print('- freezing deep layers...', CFG)
            for name, child in model.named_children():
                if name not in ['classifier', 'fc', 'head']:
                    for param in child.parameters():
                        param.requires_grad = False
            
        # get losses            
        trn_criterion, val_criterion = get_losses(CFG, device, epoch)


        ### MODELING

        # training
        gc.collect()
        if CFG['device'] == 'TPU':
            pl_loader = pl.ParallelLoader(trn_loader, [device])
        trn_loss = train_epoch(loader     = trn_loader if CFG['device'] != 'TPU' else pl_loader.per_device_loader(device), 
                               model      = model, 
                               optimizer  = optimizer, 
                               scheduler  = scheduler,
                               criterion  = trn_criterion, 
                               epoch      = epoch,
                               CFG        = CFG,
                               device     = device)

        # inference
        gc.collect()
        if CFG['device'] == 'TPU':
            pl_loader = pl.ParallelLoader(val_loader, [device])
        val_loss, val_ids, val_preds = valid_epoch(loader    = val_loader if CFG['device'] != 'TPU' else pl_loader.per_device_loader(device), 
                                                   model     = model, 
                                                   criterion = val_criterion, 
                                                   CFG       = CFG,
                                                   device    = device)
        

        ### EVALUATION
       
        # reduce preds & losses
        if CFG['device'] == 'TPU' and CFG['tpu_workers'] != 1:
            val_ids   = xm.mesh_reduce('ids',   val_ids,   np.concatenate)
            val_preds = xm.mesh_reduce('preds', val_preds, np.concatenate)
            trn_loss  = xm.mesh_reduce('loss',  trn_loss, lambda x: sum(x) / len(df_trn))
            val_loss  = xm.mesh_reduce('loss',  val_loss, lambda x: sum(x) / len(df_val))
            lr        = scheduler.state_dict()['_last_lr'][0] / xm.xrt_world_size()
        else:
            trn_loss = trn_loss / len(df_trn)
            val_loss = val_loss / len(df_val)
            lr       = scheduler.state_dict()['_last_lr'][0]
            
        # sort preds
        if CFG['device'] == 'TPU':
            val_sorted_ids = np.unique(val_ids, return_index = True)[1]
            val_preds      = val_preds[val_sorted_ids, :]

        # save LR and losses
        lrs.append(lr)
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)        
        val_metrics.append(get_score(df_val[CFG['targets']].values, val_preds)[0])
        
        # feedback
        smart_print('-- epoch {}/{} | lr = {:.6f} | trn_loss = {:.4f} | val_loss = {:.4f} | val_auc = {:.4f} | {:.2f} min'.format(
            epoch + 1, CFG['num_epochs'] + CFG['fine_tune'], lrs[epoch],
            trn_losses[epoch], val_losses[epoch], val_metrics[epoch],
            (time.time() - epoch_start) / 60), CFG)
        
        # export weights and save preds
        if val_metrics[epoch] >= max(val_metrics):
            val_preds_best = val_preds.copy()
            smart_save(model.state_dict(), CFG['out_path'] + 'weights_fold{}.pth'.format(fold), CFG)
        if CFG['save_all']:
            smart_save(model.state_dict(), CFG['out_path'] + 'weights_fold{}_epoch{}.pth'.format(fold, epoch), CFG)      
    
    return trn_losses, val_losses, val_metrics, val_preds_best



####### WRAPPER FOR TPU

def run_on_tpu(rank, CFG):
    
    # run fold
    torch.set_default_tensor_type('torch.FloatTensor')
    trn_losses, val_losses, val_metrics, val_preds_best = run_fold(fold, df_trn, df_val, CFG, model, device)

    # save results
    if rank == 0:
        
        # send metrics to neptune
        if CFG['tracking']:
            for epoch in range(CFG['num_epochs']):
                neptune.send_metric('trn_loss{}'.format(fold), trn_losses[epoch])
                neptune.send_metric('val_loss{}'.format(fold), val_losses[epoch])
                neptune.send_metric('val_auc{}'.format(fold),  val_metrics[epoch])
        
        # export performance
        np.save('trn_losses.npy',     np.array(trn_losses))
        np.save('val_losses.npy',     np.array(val_losses))
        np.save('val_metrics.npy',    np.array(val_metrics))
        np.save('val_preds_best.npy', val_preds_best)