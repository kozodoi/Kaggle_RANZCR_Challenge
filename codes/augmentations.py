####### AUGMENTATIONS

def get_augs(CFG, image_size = None, p_aug = None):

    # update epoch-based parameters
    if image_size is None:
        image_size = CFG['image_size']
    if p_aug is None:
        p_aug = CFG['p_aug']

    # normalization
    if CFG['normalize']:
        if CFG['normalize'] == 'imagenet':
            CFG['pixel_mean'] = (0.485, 0.456, 0.406)
            CFG['pixels_std'] = (0.229, 0.224, 0.225)
        elif CFG['normalize'] == 'custom':
            CFG['pixel_mean'] = (0.442, 0.511, 0.318)
            CFG['pixels_std'] = (0.233, 0.236, 0.225)
    else:
        CFG['pixel_mean'] = (0, 0, 0)
        CFG['pixels_std'] = (1, 1, 1)

    # train augmentations
    train_augs = A.Compose([A.RandomResizedCrop(height = image_size, 
                                                width  = image_size,
                                                scale  = CFG['crop_scale']),
                            A.HorizontalFlip(p = 0.5),
                            A.ShiftScaleRotate(p            = p_aug,
                                               shift_limit  = CFG['ssr'][0],
                                               scale_limit  = CFG['ssr'][1],
                                               rotate_limit = CFG['ssr'][2]),
                            A.HueSaturationValue(p               = p_aug,
                                                 hue_shift_limit = CFG['huesat'][0],
                                                 sat_shift_limit = CFG['huesat'][1],
                                                 val_shift_limit = CFG['huesat'][2]),
                            A.RandomBrightnessContrast(p                = p_aug,
                                                       brightness_limit = CFG['bricon'][0],
                                                       contrast_limit   = CFG['bricon'][1]),
                            A.OneOf([A.MotionBlur(blur_limit   = CFG['blur_limit']),
                                     A.MedianBlur(blur_limit   = CFG['blur_limit']),
                                     A.GaussianBlur(blur_limit = CFG['blur_limit'])], 
                                     p = p_aug),
                            A.OneOf([A.OpticalDistortion(distort_limit = CFG['dist_limit']),
                                     A.GridDistortion(distort_limit    = CFG['dist_limit'])], 
                                     p = p_aug),
                            A.Cutout(p          = p_aug, 
                                     num_holes  = CFG['cutout'][0], 
                                     max_h_size = np.int(CFG['cutout'][1] * image_size), 
                                     max_w_size = np.int(CFG['cutout'][1] * image_size)),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixels_std']),
                            ToTensorV2()
                           ])

    # valid augmentations
    valid_augs = A.Compose([A.Resize(height  = image_size, 
                                     width   = image_size),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixels_std']),
                            ToTensorV2()
                           ])
    
    # output
    return train_augs, valid_augs



####### TTA FLIPS
 
'''
Based on https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
'''

def get_tta_flips(img, i):

    if i >= 4:
        img = img.transpose(2, 3)
    if i % 4 == 0:
        return img
    elif i % 4 == 1:
        return img.flip(3)
    elif i % 4 == 2:
        return img.flip(2)
    elif i % 4 == 3:
        return img.flip(3).flip(2)