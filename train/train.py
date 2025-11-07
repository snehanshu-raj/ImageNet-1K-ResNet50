import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torchvision import models
from datasets import load_dataset
from tqdm import tqdm
import time

import config
from augmentations import (
    get_train_transforms, 
    get_val_transforms,
    MixupCutmix,
    mixup_criterion
)
from utils import (
    setup_logger,
    accuracy,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    reduce_tensor
)

class ImageNetCollate:
    """Collate function that can be pickled"""
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

def get_dataloaders(rank, world_size):
    """Create distributed dataloaders"""
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
    
    train_collate = ImageNetCollate(is_train=True)
    val_collate = ImageNetCollate(is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
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
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        collate_fn=val_collate,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    return train_loader, val_loader

def train_epoch(train_loader, model, criterion, optimizer, scheduler, scaler, 
                epoch, rank, world_size, mixup_cutmix=None):    
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for images, targets in pbar:
        images = images.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)
        
        use_mixup = False
        if mixup_cutmix is not None and (config.USE_MIXUP or config.USE_CUTMIX):
            images, targets_a, targets_b, lam = mixup_cutmix(images, targets)
            use_mixup = True
        
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            outputs = model(images)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        if not use_mixup:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(outputs, targets_a, topk=(1, 5))
        
        if world_size > 1:
            loss = reduce_tensor(loss.detach())
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        if rank == 0:
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{top1.avg:.2f}%',
                'Top5': f'{top5.avg:.2f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, rank, world_size):    
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
        
        for images, targets in pbar:
            images = images.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            if world_size > 1:
                loss = reduce_tensor(loss)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            if rank == 0:
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{top1.avg:.2f}%',
                    'Top5': f'{top5.avg:.2f}%'
                })
    
    return losses.avg, top1.avg, top5.avg

def train_worker(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
    logger = None
    if rank == 0:
        log_file = config.LOG_DIR / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger('train', log_file)
        logger.info("="*80)
        logger.info("IMAGENET TRAINING - RESNET50")
        logger.info("="*80)
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Batch size per GPU: {config.BATCH_SIZE_PER_GPU}")
        logger.info(f"Total batch size: {config.BATCH_SIZE_PER_GPU * world_size}")
        logger.info(f"Max LR: {config.get_scaled_lr():.6f}")
        logger.info(f"Epochs: {config.EPOCHS}")
        logger.info(f"Num workers: {config.NUM_WORKERS}")
        logger.info("="*80)
    
    if rank == 0:
        print("\nLoading ImageNet dataset...")
    train_loader, val_loader = get_dataloaders(rank, world_size)
    if rank == 0:
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
    
    if rank == 0:
        print(f"\nCreating {config.MODEL_NAME}...")
    model = models.resnet50(weights=None, num_classes=config.NUM_CLASSES)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING).to(rank)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get_scaled_lr(),
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=config.NESTEROV
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get_scaled_lr(),
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=config.ONECYCLE_PCT_START,
        div_factor=config.ONECYCLE_DIV_FACTOR,
        final_div_factor=config.ONECYCLE_FINAL_DIV_FACTOR,
        anneal_strategy='cos'
    )
    
    scaler = GradScaler(enabled=config.USE_AMP)
    
    mixup_cutmix = None
    if config.USE_MIXUP or config.USE_CUTMIX:
        mixup_cutmix = MixupCutmix(
            mixup_alpha=config.MIXUP_ALPHA,
            cutmix_alpha=config.CUTMIX_ALPHA,
            cutmix_prob=config.CUTMIX_PROB
        )
    
    start_epoch = 0
    best_acc1 = 0.0
    
    if config.RESUME:
        checkpoint_path = config.CHECKPOINT_DIR / 'checkpoint.pth'
        if checkpoint_path.exists():
            if rank == 0:
                print(f"\nResuming from {checkpoint_path}")
            start_epoch, best_acc1 = load_checkpoint(
                checkpoint_path, model.module, optimizer, scheduler, scaler
            )
            start_epoch += 1
    
    if rank == 0:
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
    
    for epoch in range(start_epoch, config.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_top1, train_top5 = train_epoch(
            train_loader, model, criterion, optimizer, scheduler, scaler,
            epoch, rank, world_size, mixup_cutmix
        )
        
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, rank, world_size)
        
        if rank == 0:
            logger.info(f"\nEpoch {epoch}/{config.EPOCHS}")
            logger.info(f"Train Loss: {train_loss:.4f} | Top1: {train_top1:.2f}% | Top5: {train_top5:.2f}%")
            logger.info(f"Val   Loss: {val_loss:.4f} | Top1: {val_top1:.2f}% | Top5: {val_top5:.2f}%")
            
            is_best = val_top1 > best_acc1
            best_acc1 = max(val_top1, best_acc1)
            
            if (epoch + 1) % config.SAVE_FREQ == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, config.CHECKPOINT_DIR)
                
                logger.info(f"Checkpoint saved! Best Top-1: {best_acc1:.2f}%")
    
    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Best Top-1 Accuracy: {best_acc1:.2f}%")
        print("="*80)
    
    dist.destroy_process_group()

def main():  
    world_size = config.WORLD_SIZE
    
    if world_size < 1:
        print("No GPUs found!")
        return
    
    print(f"Starting training on {world_size} GPU(s)")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
