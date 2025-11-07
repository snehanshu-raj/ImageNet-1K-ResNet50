import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from datasets import load_dataset
from tqdm import tqdm
import time
import logging

import fine_tune_config as config
from augmentations import get_train_transforms, get_val_transforms
from utils import accuracy, AverageMeter, save_checkpoint, reduce_tensor

class ImageNetCollate:    
    def __init__(self, is_train=True):
        self.is_train = is_train
        if is_train:
            self.transform = get_train_transforms(config)
        else:
            self.transform = get_val_transforms(config)
    
    def __call__(self, batch):
        images = []
        labels = []
        for sample in batch:
            img = sample['image'].convert('RGB')
            label = sample['label']
            images.append(self.transform(img))
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

def get_dataloaders(rank=None, world_size=None):    
    use_distributed = (rank is not None and world_size is not None and world_size > 1)
    
    if rank == 0 or rank is None:
        print("Loading ImageNet dataset...")
    
    train_dataset = load_dataset(
        "imagenet-1k",
        split="train",
        cache_dir="/project2/jieyuz_1727/snehansh/imagenet/cache",
    )
    
    val_dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir="/project2/jieyuz_1727/snehansh/imagenet/cache",
    )
    
    train_collate = ImageNetCollate(is_train=True)
    val_collate = ImageNetCollate(is_train=False)
    
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        collate_fn=train_collate,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        collate_fn=val_collate,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    if rank == 0 or rank is None:
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def create_model():
    return models.resnet50(weights=None, num_classes=config.NUM_CLASSES)

def load_pretrained_weights(model):
    """Load pretrained weights (NOT checkpoint, just weights)"""
    if config.LOAD_PRETRAINED and config.PRETRAINED_WEIGHTS:
        if os.path.exists(config.PRETRAINED_WEIGHTS):
            print(f"Loading pretrained weights: {config.PRETRAINED_WEIGHTS}")
            pretrained_dict = torch.load(config.PRETRAINED_WEIGHTS, map_location='cpu')
            model.load_state_dict(pretrained_dict)
            print("Pretrained weights loaded successfully!")
            return True
        else:
            print(f"Pretrained weights not found: {config.PRETRAINED_WEIGHTS}")
    return False


def check_checkpoint_exists():
    """Check if checkpoint exists"""
    checkpoint_path = config.CHECKPOINT_DIR / 'checkpoint.pth'
    return checkpoint_path.exists()

def load_checkpoint_full(model, optimizer, scheduler, scaler):
    checkpoint_path = config.CHECKPOINT_DIR / 'checkpoint.pth'
    
    if not checkpoint_path.exists():
        return 0, 0.0, False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
        except Exception as e:
            print(f"Could not load scheduler state: {e}")
            print("Will create fresh scheduler for remaining epochs")
    
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc1 = checkpoint.get('best_acc1', 0.0)
    
    print(f"Resumed from epoch {start_epoch-1}, Best Top-1: {best_acc1:.2f}%")
    
    return start_epoch, best_acc1, True

def get_scheduler(optimizer, start_epoch=0):

    if config.SCHEDULER_TYPE == "constant":
        return None
    
    elif config.SCHEDULER_TYPE == "constant_decay":
        # Adjust T_max for remaining epochs
        remaining_epochs = config.EPOCHS - start_epoch
        progress = start_epoch / config.EPOCHS
        cosine_factor = (1 + math.cos(math.pi * progress)) / 2
        current_lr = config.DECAY_MIN_LR + (config.LR - config.DECAY_MIN_LR) * cosine_factor
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print(f"Adjusted LR for epoch {start_epoch}: {current_lr:.6f}")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs if start_epoch > 0 else config.EPOCHS,
            eta_min=config.DECAY_MIN_LR
        )
    
    elif config.SCHEDULER_TYPE == "step":
        if start_epoch > 0:
            adjusted_milestones = [m - start_epoch for m in config.STEP_MILESTONES if m > start_epoch]
            if not adjusted_milestones:
                return None
        else:
            adjusted_milestones = config.STEP_MILESTONES
        
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=adjusted_milestones,
            gamma=config.STEP_GAMMA
        )
    
    else:
        return None

def train_epoch_single_gpu(train_loader, model, criterion, optimizer, scaler, epoch, logger):    
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(config.DEVICE, non_blocking=True)
        targets = targets.to(config.DEVICE, non_blocking=True)
        
        with autocast(enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / config.ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        losses.update((loss.item() * config.ACCUMULATION_STEPS), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Top1': f'{top1.avg:.2f}%',
            'LR': f'{current_lr:.6f}'
        })
    
    if len(train_loader) % config.ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return losses.avg, top1.avg, top5.avg


def train_epoch_multi_gpu(train_loader, model, criterion, optimizer, scaler, 
                          epoch, rank, world_size):    
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
    else:
        pbar = train_loader
    
    for images, targets in pbar:
        images = images.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)
        
        with autocast(enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        loss = reduce_tensor(loss.detach())
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{top1.avg:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device, rank=None, world_size=None):    
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    use_distributed = (rank is not None and world_size is not None and world_size > 1)
    
    with torch.no_grad():
        if rank == 0 or rank is None:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            if use_distributed:
                loss = reduce_tensor(loss)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            if rank == 0 or rank is None:
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.2f}%'
                })
    
    return losses.avg, top1.avg, top5.avg

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def train_single_gpu():    
    print("\n" + "="*80)
    print("SINGLE GPU MODE")
    print("="*80)
    print(f"Physical batch: {config.BATCH_SIZE_PER_GPU}")
    print(f"Accumulation: {config.ACCUMULATION_STEPS}")
    print(f"Effective batch: {config.BATCH_SIZE_PER_GPU * config.ACCUMULATION_STEPS}")
    print("="*80 + "\n")
    
    log_file = config.LOG_DIR / f"train_single_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger('train', log_file)
    logger.info("SINGLE GPU FINE-TUNING")
    
    train_loader, val_loader = get_dataloaders()
    
    print("Creating ResNet-50 model...")
    model = create_model()
    
    checkpoint_exists = check_checkpoint_exists()
    
    if config.RESUME and checkpoint_exists:
        print("=" * 80)
        print("CHECKPOINT FOUND - Will resume training")
        print("=" * 80)
    else:
        print("=" * 80)
        print("NO CHECKPOINT - Starting fresh")
        print("=" * 80)
        loaded_pretrained = load_pretrained_weights(model)
        if not loaded_pretrained:
            print("Training from random initialization")
    
    model = model.to(config.DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING).to(config.DEVICE)
    
    if config.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LR,
            betas=config.ADAM_BETAS,
            eps=config.ADAM_EPS,
            weight_decay=config.WEIGHT_DECAY,
            amsgrad=config.ADAM_AMSGRAD
        )
    elif config.OPTIMIZER == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.LR,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
                nesterov=config.NESTEROV
        )
    
    scaler = GradScaler(enabled=config.USE_AMP)
    
    start_epoch = 0
    best_acc1 = 0.0
    scheduler = None
    
    if config.RESUME and checkpoint_exists:
        if config.USE_SCHEDULER:
            scheduler = get_scheduler(optimizer, start_epoch=0)
        
        start_epoch, best_acc1, checkpoint_loaded = load_checkpoint_full(
            model, optimizer, scheduler, scaler
        )
        
        if checkpoint_loaded and config.USE_SCHEDULER:
            print(f"Creating scheduler for remaining {config.EPOCHS - start_epoch} epochs...")
            scheduler = get_scheduler(optimizer, start_epoch=start_epoch)
    else:
        if config.USE_SCHEDULER:
            scheduler = get_scheduler(optimizer, start_epoch=0)
    
    print("\n" + "="*80)
    print(f"STARTING TRAINING FROM EPOCH {start_epoch}")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss, train_top1, train_top5 = train_epoch_single_gpu(
            train_loader, model, criterion, optimizer, scaler, epoch, logger
        )
        
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, config.DEVICE)
        
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\nEpoch {epoch}/{config.EPOCHS}")
        logger.info(f"Train: Loss={train_loss:.4f}, Top1={train_top1:.2f}%")
        logger.info(f"Val:   Loss={val_loss:.4f}, Top1={val_top1:.2f}%")
        logger.info(f"LR: {current_lr:.6f}")
        
        is_best = val_top1 > best_acc1
        best_acc1 = max(val_top1, best_acc1)
        
        if (epoch + 1) % config.SAVE_FREQ == 0 or is_best:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, config.CHECKPOINT_DIR)
            
            if is_best:
                logger.info(f"Best: {best_acc1:.2f}%")
    
    print(f"\nDONE! Best accuracy: {best_acc1:.2f}%\n")

def train_multi_gpu_worker(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
    logger = None
    if rank == 0:
        log_file = config.LOG_DIR / f"train_multi_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger('train', log_file)
        logger.info("="*80)
        logger.info("MULTI-GPU DDP FINE-TUNING")
        logger.info("="*80)
        logger.info(f"GPUs: {world_size}")
        logger.info(f"Batch per GPU: {config.BATCH_SIZE_PER_GPU}")
        logger.info(f"Total batch: {config.BATCH_SIZE_PER_GPU * world_size}")
        logger.info(f"LR: {config.LR}")
        logger.info("="*80)
    
    train_loader, val_loader = get_dataloaders(rank, world_size)
    
    if rank == 0:
        print("\nCreating ResNet-50 model...")
    model = create_model()
    
    checkpoint_exists = check_checkpoint_exists()
    
    if config.RESUME and checkpoint_exists:
        if rank == 0:
            print("=" * 80)
            print("CHECKPOINT FOUND - Will resume training")
            print("=" * 80)
    else:
        if rank == 0:
            print("=" * 80)
            print("NO CHECKPOINT - Starting fresh")
            print("=" * 80)
        loaded_pretrained = load_pretrained_weights(model)
        if not loaded_pretrained and rank == 0:
            print("Training from random initialization")
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    if rank == 0:
        print(f"Model wrapped with DDP")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING).to(rank)
    
    if config.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LR,
            betas=config.ADAM_BETAS,
            eps=config.ADAM_EPS,
            weight_decay=config.WEIGHT_DECAY,
            amsgrad=config.ADAM_AMSGRAD
        )
    elif config.OPTIMIZER == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.LR,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
                nesterov=config.NESTEROV
        )
    
    scaler = GradScaler(enabled=config.USE_AMP)
    
    start_epoch = 0
    best_acc1 = 0.0
    scheduler = None
    
    if config.RESUME and checkpoint_exists:
        if config.USE_SCHEDULER:
            scheduler = get_scheduler(optimizer, start_epoch=0)
        
        start_epoch, best_acc1, checkpoint_loaded = load_checkpoint_full(
            model, optimizer, scheduler, scaler
        )
        
        if checkpoint_loaded and config.USE_SCHEDULER:
            if rank == 0:
                print(f"Creating scheduler for remaining {config.EPOCHS - start_epoch} epochs...")
            scheduler = get_scheduler(optimizer, start_epoch=start_epoch)
    else:
        if config.USE_SCHEDULER:
            scheduler = get_scheduler(optimizer, start_epoch=0)
    
    if rank == 0:
        print("\n" + "="*80)
        print(f"STARTING TRAINING FROM EPOCH {start_epoch}")
        print("="*80 + "\n")
    
    for epoch in range(start_epoch, config.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_top1, train_top5 = train_epoch_multi_gpu(
            train_loader, model, criterion, optimizer, scaler,
            epoch, rank, world_size
        )
        
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, rank, rank, world_size)
        
        if scheduler:
            scheduler.step()
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"\nEpoch {epoch}/{config.EPOCHS}")
            logger.info(f"Train: Loss={train_loss:.4f}, Top1={train_top1:.2f}%")
            logger.info(f"Val:   Loss={val_loss:.4f}, Top1={val_top1:.2f}%")
            logger.info(f"LR: {current_lr:.6f}")
            
            is_best = val_top1 > best_acc1
            best_acc1 = max(val_top1, best_acc1)
            
            if (epoch + 1) % config.SAVE_FREQ == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, config.CHECKPOINT_DIR)
                
                if is_best:
                    logger.info(f"Best: {best_acc1:.2f}%")
    
    if rank == 0:
        print(f"\nDONE! Best accuracy: {best_acc1:.2f}%\n")
    
    dist.destroy_process_group()

def main():    
    num_gpus = torch.cuda.device_count()
    
    print("\n" + "="*80)
    print("UNIVERSAL FINE-TUNING SCRIPT")
    print("="*80)
    print(f"Detected GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs found!")
        return
    
    elif num_gpus == 1:
        print("Mode: Single GPU with gradient accumulation")
        print("="*80)
        train_single_gpu()
    
    else:
        print(f"Mode: Multi-GPU DDP ({num_gpus} GPUs)")
        print("="*80)
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        mp.spawn(
            train_multi_gpu_worker,
            args=(num_gpus,),
            nprocs=num_gpus,
            join=True
        )


if __name__ == "__main__":
    main()
