import os
import random
import shutil

SOURCE = "recipe_dataset"

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(f"dataset/{split}", exist_ok=True)

for cls in os.listdir(SOURCE):

    cls_path = os.path.join(SOURCE, cls)

    if not os.path.isdir(cls_path):
        continue

    files = os.listdir(cls_path)

    random.shuffle(files)

    train_end = int(0.7 * len(files))
    val_end = int(0.85 * len(files))

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    for split, group in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:

        out_dir = f"dataset/{split}/{cls}"

        os.makedirs(out_dir, exist_ok=True)

        for f in group:
            shutil.copy(
                os.path.join(cls_path, f),
                os.path.join(out_dir, f)
            )

print("Done")
