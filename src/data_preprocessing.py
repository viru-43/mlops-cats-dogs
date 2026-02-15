import os
import shutil
import random
from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw/PetImages")
PROCESSED_DIR = Path("data/processed")

IMG_SIZE = (224, 224)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

MAX_IMAGES_PER_CLASS = 3000  # reduce for CPU training speed


def clean_and_collect_images(class_name):
    class_path = RAW_DIR / class_name
    images = []

    for img_file in class_path.glob("*.jpg"):
        try:
            img = Image.open(img_file).convert("RGB")
            images.append(img_file)
        except Exception:
            continue

    random.shuffle(images)
    return images[:MAX_IMAGES_PER_CLASS]


def split_data(images):
    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    return (
        images[:train_end],
        images[train_end:val_end],
        images[val_end:]
    )


def save_images(image_paths, split, class_name):
    save_dir = PROCESSED_DIR / split / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(save_dir / img_path.name)
        except Exception:
            continue


def main():
    random.seed(42)

    for class_name in ["Cat", "Dog"]:
        print(f"Processing {class_name}...")

        images = clean_and_collect_images(class_name)
        train, val, test = split_data(images)

        save_images(train, "train", class_name)
        save_images(val, "val", class_name)
        save_images(test, "test", class_name)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
