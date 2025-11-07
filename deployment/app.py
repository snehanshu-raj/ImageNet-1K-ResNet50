import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import os

def load_imagenet_classes():
    try:
        with open('imagenet_label_map.json', 'r') as f:
            label_map = json.load(f)
        
        class_names = []
        for i in range(1000):
            class_names.append(label_map[str(i)])
        
        print(f"Loaded {len(class_names)} class names from imagenet_label_map.json")
        return class_names
    
    except Exception as e:
        print(f"Error loading label map: {e}")
        print("Using fallback class names")
        return [f"class_{i}" for i in range(1000)]


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(checkpoint_path='model.pth', device='cpu'):
    print(f"Loading model from {checkpoint_path}...")
    model = models.resnet50(weights=None, num_classes=1000)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    best_acc = checkpoint.get('best_acc1', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    
    print(f"Model loaded.")
    print(f"   Best training accuracy: {best_acc}%")
    print(f"   Trained for: 90 (scratch) + {epoch} (fine-tuned) epochs")
    
    return model, best_acc, epoch

print("Initializing ImageNet Classifier...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

model, best_acc, trained_epoch = load_model(device=device)
transform = get_transform()
class_names = load_imagenet_classes()

print("Ready for inference!")

def classify_image(image):
    if image is None:
        return {}
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
    
    top_probs, top_indices = torch.topk(probabilities, 5)
    
    results = {}
    for prob, idx in zip(top_probs, top_indices):
        class_idx = idx.item()
        class_name = class_names[class_idx]
        results[class_name] = float(prob.item())
    
    return results

custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
.gr-button {
    color: white;
    border-color: black;
    background: black;
}
.gr-button:hover {
    background: #1f1f1f;
}
footer {
    display: none !important;
}
"""

acc_display = f"{best_acc:.2f}" if isinstance(best_acc, (int, float)) else "70.0"
epoch_display = f"{trained_epoch}" if isinstance(trained_epoch, int) else "90"

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        f"""
        # ImageNet ResNet-50 Classifier
        
        Upload an image to classify it into one of 1000 ImageNet categories.
        
        ## Model Details
        - **Architecture**: ResNet-50 (25.5M parameters)
        - **Training**: From scratch on ImageNet-1K (1.28M images)
        - **Best Accuracy**: {acc_display}% Top-1
        - **Augmentations**: RandomResizedCrop, ColorJitter, AutoAugment, Mixup, CutMix, Random Erasing
        - **Training Setup**: 3 GPUs, batch size 2,688, OneCycleLR scheduler
        - **Fine-Tuning Setup**: 2 GPUs, batch size 1280, Cosine Decay
        
        *Uses exact label mapping from Hugging Face ImageNet-1K dataset.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            
            classify_btn = gr.Button("Classify Image", variant="primary", size="lg")
            clear_btn = gr.ClearButton([image_input], value="Clear")
            
            gr.Markdown(
                """
                ### 💡 Tips
                - Upload clear, well-lit images
                - Works best with centered objects
                - Supports common formats (JPG, PNG, etc.)
                - Images will be resized to 224×224
                """
            )
        
        with gr.Column(scale=1):
            label_output = gr.Label(
                label="🏆 Top 5 Predictions",
                num_top_classes=5
            )
            
            gr.Markdown(
                f"""
                ### Model Statistics
                
                **Training Configuration:**
                - Optimizer: SGD with Nesterov momentum (0.9)
                - Learning Rate: OneCycleLR (max: 0.17, warmup: 12%)
                - Weight Decay: 1e-4
                - Label Smoothing: 0.5
                - Batch Size: 2,688 total (896 per GPU × 3 GPUs)
                
                **Inference Pipeline:**
                1. Resize to 256×256
                2. Center crop to 224×224
                3. Normalize with ImageNet statistics
                4. Forward pass through ResNet-50
                5. Softmax to get probabilities
                
                **Current Status:** Model loaded on **{device.upper()}**
                """
            )
            
    example_images = [[os.path.join("samples", f)] for f in os.listdir("samples") if f.endswith(".jpg")]
    gr.Markdown("### Try Example Images")
    gr.Examples(
        examples=example_images,
        inputs=image_input,
        outputs=label_output,
        fn=classify_image,
        cache_examples=False,
        label="Click to try:"
    )
    
    classify_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=label_output
    )
    
    gr.Markdown(
        """
        ---
        **Built with ❤️ using PyTorch and Gradio** | 
        Model trained from scratch on ImageNet-1K using Hugging Face Datasets | 
        Training configuration: OneCycleLR, heavy augmentation, distributed training
        """
    )
    
if __name__ == "__main__":
    demo.launch(
        share=True,  # for temporary public link
        server_name="0.0.0.0",  
        server_port=7860  
    )
