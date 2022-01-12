# %%

from typing import List
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input
from tensorflow.math import confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from tensorflow.data import Dataset
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.autograph.experimental import do_not_convert

from PIL import Image
import numpy as np
import plotly.express as px


class Pipeline:

    train_dir: str
    test_dir : str

    train_data: Dataset
    val_data  : Dataset
    test_data : Dataset

    rescale_layer: Sequential
    augmentation : Sequential

    classes_names: List[str]
    num_classes  : int

    image_size : tuple
    input_shape: tuple
    
    
    def __init__(self, train_dir: str, test_dir: str) -> None:

        self.train_dir = train_dir
        self.test_dir  = test_dir

        self.rescale_layer = Sequential(layers.Rescaling(1./255))
        self.augmentation  = Sequential(layers.RandomFlip("horizontal"))

        self.classes_names = os.listdir(train_dir)

        self.get_data()
        self._infer_input_shape()


    def _donwload_data(self) -> None:
        
        print("Downloading datasets")

        data = fashion_mnist.load_data()
        train_data, test_data = data

        self.num_classes = len(np.unique([label for label in test_data[1]]))
        self.train_data  = train_data
        self.test_data   = test_data
    
    
    def _save_data(self) -> None:

        print("Saving the dataset to disk")

        labels = {0: "T-shirt",
                  1: "Trouser",
                  2: "Pullover",
                  3: "Dress",
                  4: "Coat",
                  5: "Sandal",
                  6: "Shirt",
                  7: "Sneaker",
                  8: "Bag",
                  9: "Ankle boot"}
                  
        suffix = {i: 0 for i in range(self.num_classes)}

        inputs, targets = self.train_data
        
        for img, label in zip(inputs, targets):
            
            image     = Image.fromarray(img)
            image_dir = f"{self.train_dir}/{labels[label]}/"

            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            
            image.save(f"{image_dir}/image_{suffix[label]}.png")        
            suffix[label] += 1

        suffix = {i: 0 for i in range(self.num_classes)}

        inputs, targets = self.test_data
        
        for img, label in zip(inputs, targets):
            
            image     = Image.fromarray(img)
            image_dir = f"{self.test_dir}/{labels[label]}/"
            
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            
            image.save(f"{image_dir}/image_{suffix[label]}.png")
            suffix[label] += 1


    def _infer_input_shape(self):

        item = self.test_data.take(1)
        image, label = item.get_single_element()

        self.input_shape = image.numpy().shape[1: ]


    def _infer_image_size(self, set: str):

        folder_path = set + f"/{self.classes_names[0]}"
        image_path  = os.listdir(folder_path)[0]
        image       = Image.open(folder_path + f"/{image_path}")

        return image.size


    def _load_data(self) -> None:
        
        print("Fetching datasets")

        train_image_size = self._infer_image_size(set = self.train_dir)

        train_ds = image_dataset_from_directory(self.train_dir,
                                                image_size = train_image_size,
                                                batch_size = 128,
                                                color_mode = "grayscale",
                                                subset = "training",
                                                seed = 1,
                                                validation_split = 0.2)

        train_ds = train_ds.map(do_not_convert(lambda x, y: (self.rescale_layer(x), y)), num_parallel_calls = AUTOTUNE)
        train_ds = train_ds.shuffle(len(train_ds)).prefetch(buffer_size = AUTOTUNE).cache()

        val_ds = image_dataset_from_directory(self.train_dir,
                                              image_size = train_image_size,
                                              batch_size = 128,
                                              color_mode = "grayscale",
                                              subset = "validation",
                                              seed = 1,
                                              validation_split = 0.2)

        val_ds = val_ds.map(do_not_convert(lambda x, y: (self.rescale_layer(x), y)), num_parallel_calls = AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size = AUTOTUNE).cache()
        
        train_image_size = self._infer_image_size(set = self.train_dir)
        
        test_ds = image_dataset_from_directory(self.test_dir,
                                               batch_size = 1,
                                               image_size = train_image_size,
                                               color_mode = "grayscale")

        self.train_data = train_ds
        self.val_data   = val_ds
        self.test_data  = test_ds


    def get_data(self) -> None:

        if not os.path.isdir(self.train_dir):
            self._donwload_data()
            self._save_data()
            self._load_data()
        else:
            
            self._load_data()
            self.num_classes = 10


# %%

train_dir = "C:/Users/Joao/Kotoamatsukami/Kotoamatsukami/data/fashion_mnist/train"
test_dir  = "C:/Users/Joao/Kotoamatsukami/Kotoamatsukami/data/fashion_mnist/test"
pipeline  = Pipeline(train_dir = train_dir, test_dir = test_dir)


# %%

input_shape = pipeline.input_shape
num_classes = pipeline.num_classes

model = Sequential()
model.add(Input(shape = input_shape))
model.add(Conv2D(32, kernel_size = (3, 3), activation = "swish"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = "swish"))
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Conv2D(16, kernel_size = (3, 3), activation = "swish"))
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Flatten())
model.add(Dense(256, activation = "swish"))
model.add(Dense(128, activation = "swish"))
model.add(Dense(128, activation = "swish"))
model.add(Dense(num_classes, activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# %%

history = model.fit(pipeline.train_data,
                    epochs = 10,
                    validation_data = pipeline.val_data)

# %%

inputs  = Input(shape = (28, 28, 1))
x       = pipeline.rescale_layer(inputs)
outputs = model(x)

trained_model = Model(inputs, outputs)

# trained_model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# %%

loss, acc = trained_model.evaluate(pipeline.test_data)


# %%

truth       = []
predictions = []

for image, label in pipeline.test_data:
    predictions.append(trained_model.predict(image))
    truth.append(label)

preds = [np.argmax(p) for p in predictions]

confusion = confusion_matrix(labels = truth, predictions = preds, num_classes = pipeline.num_classes)

fig = px.imshow(confusion, color_continuous_scale = "viridis")
fig.show()


# %%

from numba import jit

def get_confusion_matrix(test_data, model, num_classes):

    truth       = [items[1].numpy()[0] for items in test_data]
    predictions = model.predict(test_data)

    preds = [np.argmax(p) for p in predictions]

    confusion = confusion_matrix(labels = truth, predictions = preds, num_classes = num_classes)

    fig = px.imshow(confusion, color_continuous_scale = "viridis")
    fig.show()

# %%

get_confusion_matrix(pipeline.test_data, trained_model, pipeline.num_classes)