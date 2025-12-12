import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model('animal_classifier_finetuned.keras')

# Load class names (assuming you saved them, else manually define)
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']  # Example

def load_image(path):
    img = Image.open(path).convert('RGB')
    width, height = img.size
    min_side = min(width, height)

    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    cropped = img.crop((left, top, right, bottom))

    resized = cropped.resize((256, 256), Image.LANCZOS)
    arr = np.asarray(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    plt.imshow(resized, cmap='gray')
    plt.title("Processed 256x256 Input")
    plt.axis('off')
    plt.show()

    return arr, resized

image_arr, img = load_image("img.png")

predictions = model.predict(image_arr)
predicted_index= np.argmax(predictions)
predicted_label = class_names[predicted_index]

# Display
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

# Ask for feedback
correct_label = input("Enter correct label (or press Enter to confirm): ").strip().lower()

# If user corrects it, save image for future training
if correct_label and correct_label != predicted_label:
    save_dir = os.path.join("confirmed_data", correct_label)
    os.makedirs(save_dir, exist_ok=True)

    # Find a unique filename
    base_name = "correction"
    ext = ".png"
    counter = 1
    while os.path.exists(os.path.join(save_dir, f"{base_name}_{counter}{ext}")):
        counter += 1
    filename = f"{base_name}_{counter}{ext}"

    img.save(os.path.join(save_dir, filename))
    print(f"Saved corrected image to {os.path.join(save_dir, filename)}")