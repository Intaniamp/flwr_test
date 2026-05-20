import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

source_dir = os.path.join(ROOT_DIR, "dataset")
train_dir = os.path.join(ROOT_DIR, "dataset_split", "train")
val_dir = os.path.join(ROOT_DIR, "dataset_split", "val")

print("Mencari dataset di:", source_dir)
if not os.path.exists(source_dir):
    raise FileNotFoundError(f"Folder tidak ditemukan: {source_dir}")

split_ratio = 0.8

# Bersihkan folder lama kalau sudah ada
shutil.rmtree(train_dir, ignore_errors=True)
shutil.rmtree(val_dir, ignore_errors=True)

random.seed(42)

# Looping tiap class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

    print(f"✅ {class_name}: train={len(train_images)}, val={len(val_images)}")

print("\n🎉 Split dataset sukses! Cek folder 'dataset_split' di FLWR-TEST")