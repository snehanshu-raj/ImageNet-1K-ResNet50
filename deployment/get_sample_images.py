from datasets import load_dataset
from PIL import Image
from pathlib import Path
import random
from datasets import load_dataset_builder

def get_imagenet_samples(num_samples=5, output_dir="samples"):
    print("Loading ImageNet-1K")
    dataset = load_dataset(
        "imagenet-1k",
        split="train",
        cache_dir="/project2/jieyuz_1727/snehansh/imagenet/cache"
    )

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    builder = load_dataset_builder("imagenet-1k")
    label_names = builder.info.features["label"].names

    print(f"Collecting {num_samples} random samples...")

    iterator = iter(dataset)
    samples = []
    for _ in range(num_samples * 5):  
        sample = next(iterator)
        samples.append(sample)

    chosen = random.sample(samples, num_samples)

    for i, sample in enumerate(chosen):
        img: Image.Image = sample["image"]
        label_id = sample["label"]
        label_name = label_names[label_id]

        if img.mode != "RGB":
            img = img.convert("RGB")

        filename = f"sample_{i+1}_{label_name.replace(' ', '_')}.jpg"
        path = Path(output_dir) / filename
        img.save(path, quality=95)

        print(f"Saved {filename} ({label_name})")

    print(f"\nDone! Saved {num_samples} images in '{output_dir}/'.")

if __name__ == "__main__":
    get_imagenet_samples(num_samples=5, output_dir="samples")
