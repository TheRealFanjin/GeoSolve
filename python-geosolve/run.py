import keras
import tensorflow as tf

model = keras.models.load_model('saved_models/model_batch_00244.keras')
img_bytes = tf.io.read_file('9999.png')
img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, (224, 224))
# Add a batch dimension, e.g., (224, 224, 3) -> (1, 224, 224, 3)
img = tf.expand_dims(img, axis=0)
print(model.predict(img))