from tensorflow import keras
import tensorflow as tf
import absl.logging
import os


K = keras.backend
layers = keras.layers

absl.logging.set_verbosity(absl.logging.ERROR)
physical_devices = tf.config.list_physical_devices('GPU')
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


def FACEMASK01(input_shape, output_shape):

    model = keras.models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(output_shape, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'], run_eagerly=False)

    return model






