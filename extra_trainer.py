import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout



# Get the path to the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "confirmed_data")

height = 256
width = 256
batch_size = 16
seed = 42

train_new = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=seed,
    image_size=(height, width),
    batch_size=batch_size
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(height, width, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 animals
])

model.load_weights("animal_classifier_finetuned.keras")

for layer in model.layers[:-1]:  # freeze all but the last
    layer.trainable = False

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_new, epochs=1)

model.save('animal_classifier_finetuned.keras')


