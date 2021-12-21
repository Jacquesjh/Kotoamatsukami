
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input
from tensorflow.math import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.math import confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf

import numpy as np


def rescale() -> Sequential:

    rescale_layer = Sequential(layers.Rescaling(1./255))

    return rescale_layer


def data_aug() -> Sequential:

    data_augmentation = Sequential(layers.RandomFlip("horizontal"))

    return data_augmentation



def architecture() -> Sequential:

    num_classes = 10

    model = Sequential()

    model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2))) 
    model.add(Conv2D(16, kernel_size = (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2))) 
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(num_classes, activation = "softmax"))

    return model


def unify_model(rescale: Sequential, augmentation: Sequential, architecture: Sequential) -> Sequential:
    
    input_shape   = (28, 28, 1)
    inputs        = Input(shape = input_shape)
    rescale_layer = rescale(inputs)
    outputs       = architecture(inputs)
    model         = Model(inputs = inputs, outputs = outputs)

    return model


def main():
    data = fashion_mnist.load_data()
    train_data, test_data = data

    train_inputs, train_targets = train_data
    test_inputs, test_targets   = test_data

    num_channels = 1
    num_classes  = len(np.unique(train_targets))
    input_shape  = train_inputs[0].shape[0], train_inputs[0].shape[1], num_channels
    
    model = unify_model(rescale(), data_aug(), architecture())

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    history = model.fit(train_inputs,
                        train_targets,
                        batch_size = 128,
                        epochs = 5,
                        verbose = 1)

    predictions = model.predict(test_inputs)

    # preds = [np.argmax(p) for p in predictions]

    # confusion = confusion_matrix(labels = test_targets, predictions = preds, num_classes = num_classes)

    # fig = px.imshow(confusion, color_continuous_scale = "viridis")
    # fig.show()

if __name__ == "__main__":

    with tf.device("/cpu:0"):
        main()

    with tf.device("/gpu:0"):
        main()