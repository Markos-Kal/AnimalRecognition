import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Get the path to the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to your dataset
data_dir = os.path.join(base_dir, "animals10", "versions", "2", "raw-img")

sns.set(style="whitegrid")

animalCount = {}
sizes = []

for animal in os.listdir(data_dir):
    folder = os.path.join(data_dir, animal)
    if(os.path.isdir(folder)):
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        animalCount[animal] = len(files)
        for file in files:
            try:
                with Image.open(os.path.join(folder, file)) as img:
                    sizes.append(img.size)
            except:
                continue

plt.figure(figsize=(10, 5))
sns.barplot(x=list(animalCount.keys()), y=list(animalCount.values()), palette="Set2")
plt.xticks(rotation = 45)
plt.title("Images per Animal")
plt.tight_layout()
plt.show()

# Plot width distribution
widths = [w for w, h in sizes]
plt.figure(figsize=(8, 4))
sns.histplot(widths, bins=30, kde=True, color='blue')
plt.title("Width Distribution")
plt.xlabel("Width (px)")
plt.ylabel("Count")
plt.show()

# Plot height distribution
heights = [h for w, h in sizes]
plt.figure(figsize=(8, 4))
sns.histplot(heights, bins=30, kde=True, color='orange')
plt.title("Height Distribution")
plt.xlabel("Height (px)")
plt.ylabel("Count")
plt.show()

# Most common image sizes
print("Most common image sizes:")
for (w, h), c in Counter(sizes).most_common(10):
    print(f"{w}x{h} → {c} images")