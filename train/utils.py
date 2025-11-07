import torch
import torch.distributed as dist
from pathlib import Path
import logging


def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(console)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(file_handler)
    
    return logger


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


class AverageMeter:  
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    checkpoint_path = Path(checkpoint_dir) / filename
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = Path(checkpoint_dir) / 'model_best.pth'
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc1', 0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
