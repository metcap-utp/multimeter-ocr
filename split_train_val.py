import os
import shutil
import random

imgs = [
    f
    for f in os.listdir("dataset/images")
    if f.endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(imgs)

split = int(len(imgs) * 0.8)
train_imgs = imgs[:split]
val_imgs = imgs[split:]

for folder in ["train", "val"]:
    os.makedirs(f"dataset/images/{folder}", exist_ok=True)
    os.makedirs(f"dataset/labels/{folder}", exist_ok=True)

for img in train_imgs:
    name = os.path.splitext(img)[0]
    shutil.copy(f"dataset/images/{img}", f"dataset/images/train/{img}")
    shutil.copy(
        f"dataset/labels/{name}.txt", f"dataset/labels/train/{name}.txt"
    )

for img in val_imgs:
    name = os.path.splitext(img)[0]
    shutil.copy(f"dataset/images/{img}", f"dataset/images/val/{img}")
    shutil.copy(f"dataset/labels/{name}.txt", f"dataset/labels/val/{name}.txt")
