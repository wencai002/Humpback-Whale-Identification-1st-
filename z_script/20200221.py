from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
tf.image.decode_jpeg()

model = MobileNet(input_shape=(224,224,3))
embeded_train = model.predict(train_ds)