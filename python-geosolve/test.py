import zipfile as zp
import tensorflow as tf
x_train = []
path = "dataset/archive.zip/dataset"
with zp.Zipfile(path, "r") as dataset:
    for info in dataset.infolist():
        with dataset.open(info.filename) as currFile:
            data = currFile.read()
            if info.filename.endswith('.png','.jpg','jpeg'):
                image = tf.image.decode_image(data,channels=3)
                image = tf.image.convert_image_dtype(image,tf.float32)
                x_train.append(image)
            elif info.filename.endswith('.csv'):
                coords = 0
