import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

label_translation = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}

# Get the path to the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to your dataset
data_dir = os.path.join(base_dir, "animals10", "versions", "2", "raw-img")

for orig, tran in label_translation.items():
    src = os.path.join(data_dir, orig)
    dst = os.path.join(data_dir, tran)

    if os.path.exists(src):
        if not os.path.exists(dst):  # prevent overwrite or errors
            os.rename(src, dst)
            print(f"Renamed {src} → {dst}")
        else:
            print(f"Skipped {src}: destination {dst} already exists.")
    else:
        print(f"Source {src} does not exist.")