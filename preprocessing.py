import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = "data"
output_dir = "dataset"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    
    images = os.listdir(cls_path)
    train, test = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    for split, split_data in zip(["train", "val", "test"], [train, val, test]):
        split_path = os.path.join(output_dir, split, cls)
        os.makedirs(split_path, exist_ok=True)
        for img in split_data:
            shutil.copy(os.path.join(cls_path, img), os.path.join(split_path, img))

print("✅ Dataset split into train/val/test successfully!")
