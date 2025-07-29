import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import metrics
from keras import optimizers
from keras import losses
from keras import layers
import zipfile as zp
import numpy as np
import pandas as pd
import os
import pathlib
import io
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

#config
INTERNAL_DATA = "../../random-street-view/data/street_view_data"
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 50
SHUFFLE_BUFFER_SIZE = 1000
AUTOTUNE = tf.data.AUTOTUNE

MODEL_SAVE_PATH = "saved_models"
MODEL_FILENAME_FORMAT = os.path.join(MODEL_SAVE_PATH, "model_epoch_{epoch:02d}.keras")


print('loading dataset')
countries = np.array(sorted([item.name for item in pathlib.Path(INTERNAL_DATA + "batch2.jsonl").glob('*')]))


def preprocess(path):
    # get encoded country label
    country = tf.argmax(tf.strings.split(path, os.path.sep)[-3] == countries)

    # decode img
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    return image, country


# Load dataset
train_ds = tf.data.Dataset.list_files(INTERNAL_DATA + "/street_view_images", shuffle=False)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
for image, label in train_ds.take(1):
    print(image.shape)
train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


print('loading model')
model = keras.Sequential(layers=[
    keras.Input(shape=IMG_SIZE + (3,)),
    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.GlobalAvgPool2D(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(94, activation="softmax")
])
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

checkpoint_callback_n_steps = ModelCheckpoint(
     filepath=os.path.join(MODEL_SAVE_PATH, "model_batch_{batch:05d}.keras"),
     save_freq=1000,
     save_weights_only=False,
     verbose=1
 )
print('training')
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback_n_steps]
    )

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "final_trained_model")
os.makedirs(FINAL_MODEL_PATH, exist_ok=True)
model.save(FINAL_MODEL_PATH + '/model.keras')

loss = history.history["loss"]
epochsRange = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochsRange, loss, label="Training loss", color="blue")
plt.title("Training Over Loss Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (Mean Squared Error)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()