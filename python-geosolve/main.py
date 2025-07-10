import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import metrics
from keras import optimizers
from keras import losses
import zipfile as zp
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

#config
MAIN_ZIP = "dataset/archive.zip"
INTERNAL_DATA = "dataset/"
CSV_PATH = INTERNAL_DATA + "coords.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
BUFFER_SIZE = 1024

MODEL_SAVE_PATH = "saved_models"
MODEL_FILENAME_FORMAT = os.path.join(MODEL_SAVE_PATH, "model_epoch_{epoch:02d}.keras")

with zp.ZipFile(MAIN_ZIP,'r') as zf:
    with zf.open(CSV_PATH) as csvBytes:
        data = io.StringIO(csvBytes.read().decode("utf-8"))
        setOfCoords = pd.read_csv(data, header=None)
        setOfCoords.columns = ["longitude","latitude"]

imageLabelMap = {}
for index, row in setOfCoords.iterrows():
    key = INTERNAL_DATA + f"{index}.png"
    value = np.array([row["longitude"], row["latitude"]], dtype = np.float32)
    imageLabelMap[key] = value

def img_data_generator():
    with zp.ZipFile(MAIN_ZIP, 'r') as zf:
        for imgFileName, coords in imageLabelMap.items():
            with zf.open(imgFileName) as imgFile:
                imgBytes = imgFile.read()
                yield imgBytes, coords

def preprocess_img(imgBytes, coords):
    img = tf.image.decode_image(imgBytes,channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return img, coords

def preprocess_img_training(imgBytes, coords):
    img, coords = preprocess_img(imgBytes,coords)
    #Data augmentation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, coords

def training_dataset_pipeline():
    dataset = tf.data.Dataset.from_generator(
        lambda: img_data_generator(),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    dataset = dataset.map(preprocess_img_training, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

trainDataset = training_dataset_pipeline()

model = keras.Sequential(layers=[
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=IMG_SIZE + (3, )),
    keras.layers.Conv2D(32,(3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64,(3, 3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(2, activation="linear")
])
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.MeanSquaredError(),
    metrics=[
        metrics.MeanSquaredError()
    ]
)

checkpoint_callback_n_steps = ModelCheckpoint(
     filepath=os.path.join(MODEL_SAVE_PATH, "model_batch_{batch:05d}.keras"),
     save_freq=1000,
     save_weights_only=False,
     verbose=1
 )

history = model.fit(
    trainDataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback_n_steps]
    )

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "final_trained_model")
os.makedirs(FINAL_MODEL_PATH, exist_ok=True)
model.save(FINAL_MODEL_PATH)

loss = history.history["loss"]
epochsRange = range(1, EPOCHS + 1)
plt.figure(figsize = (10, 6))
plt.plot(epochsRange, loss, label="Training loss", color="blue")
plt.title("Training Over Loss Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (Mean Squared Error)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()