import tensorflow as tf

# LOAD MODEL H5
model = tf.keras.models.load_model(
    "model_bisindo_fixed.h5",
    compile=False
)

# CONVERT KE TFLITE
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# SAVE FILE
with open("model_bisindo.tflite", "wb") as f:
    f.write(tflite_model)

print("Convert berhasil")