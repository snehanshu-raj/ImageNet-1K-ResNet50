import torch
import numpy as np
from torchvision import transforms
import random

class Mixup:   
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_batch, targets_a, targets_b, lam

class CutMix:    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        _, _, H, W = batch.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        targets_a, targets_b = targets, targets[index]
        
        return batch, targets_a, targets_b, lam

class MixupCutmix:   
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5):
        self.mixup = Mixup(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha)
        self.cutmix_prob = cutmix_prob
    
    def __call__(self, batch, targets):
        if random.random() < self.cutmix_prob:
            return self.cutmix(batch, targets)
        else:
            return self.mixup(batch, targets)

def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)

def get_train_transforms(config):   
    transform_list = []
    
    if config.USE_RANDOM_RESIZED_CROP:
        transform_list.append(
            transforms.RandomResizedCrop(
                config.IMG_SIZE,
                scale=(0.08, 1.0),
                ratio=(3./4., 4./3.)
            )
        )
    else:
        transform_list.extend([
            transforms.Resize(config.RESIZE_SIZE),
            transforms.RandomCrop(config.IMG_SIZE)
        ])
    
    if config.USE_HORIZONTAL_FLIP:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if config.USE_COLOR_JITTER:
        transform_list.append(
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE
            )
        )
    
    if config.USE_AUTOAUGMENT:
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        transform_list.append(AutoAugment(policy=AutoAugmentPolicy.IMAGENET))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
    
    if config.USE_RANDOM_ERASING:
        transform_list.append(
            transforms.RandomErasing(
                p=config.RANDOM_ERASING_PROB,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random'
            )
        )
    
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
