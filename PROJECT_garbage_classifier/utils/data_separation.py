import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.seed(42) 

    classes = os.listdir(source_dir)
    for cls in classes:
        src_folder = os.path.join(source_dir, cls)
        all_files = os.listdir(src_folder)
        random.shuffle(all_files)

        total = len(all_files)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)

        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }

        for split, files in splits.items():
            dest_dir = os.path.join(target_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in files:
                src_path = os.path.join(src_folder, fname)
                dst_path = os.path.join(dest_dir, fname)
                shutil.copy2(src_path, dst_path)

    print("Разделение завершено.")


split_dataset(
    source_dir='PROJECT_garbage_classifier/TrashType_Image_Dataset',
    target_dir='PROJECT_garbage_classifier/data',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
