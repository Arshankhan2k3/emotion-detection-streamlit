import tensorflow as tf
from keras.models import load_model

# Load the Keras HDF5 model
model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)

# Convert H5 model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow optimizations (optional, improves FPS)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save tflite model
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion successful! Saved as emotion_model.tflite")
