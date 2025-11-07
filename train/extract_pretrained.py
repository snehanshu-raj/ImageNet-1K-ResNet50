import torch
import os
from pathlib import Path

def extract_pretrained_weights(checkpoint_path, output_path):
    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint
    
    if list(model_state.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from DDP wrapper...")
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    epoch = checkpoint.get('epoch', 'unknown')
    accuracy = checkpoint.get('best_acc1', 'unknown')
    
    print(f"Checkpoint info:")
    print(f"   Epoch: {epoch}")
    print(f"   Accuracy: {accuracy}%")
    print(f"   Parameters: {len(model_state)} tensors")
    
    torch.save(model_state, output_path)
    print(f"Saved pretrained weights to: {output_path}")
    print(f"Ready for fine-tuning!")

if __name__ == "__main__":
    CHECKPOINT_PATH = "/project2/jieyuz_1727/snehansh/outputs_finetune_multi/checkpoints/model_best.pth"
    OUTPUT_PATH = "/project2/jieyuz_1727/snehansh/outputs_finetune_multi/checkpoints/pretrained.pth"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("Update CHECKPOINT_PATH in this script!")
    else:
        extract_pretrained_weights(CHECKPOINT_PATH, OUTPUT_PATH)
