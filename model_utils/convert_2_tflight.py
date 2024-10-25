import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model(r'Mask_Detection\model_files\models')

# Convert the model to TFLite format
converter =tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(r'Mask_Detection\model_files\tf-light\saved_model.tflite', 'wb') as f:
    f.write(tflite_model)
