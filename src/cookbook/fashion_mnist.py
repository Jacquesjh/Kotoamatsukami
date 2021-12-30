# %%
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input
from tensorflow.math import confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.data import Dataset
from tensorflow.keras.backend import clear_session
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory

from PIL import Image
import numpy as np
import plotly.express as px


class CV:

    model        : Sequential
    rescale_layer: Sequential

    train_data: Dataset
    val_data  : Dataset
    test_data : Dataset

    num_classes: int

    train_dir: str
    test_dir : str


    def __init__(self, train_dir: str, test_dir: str) -> None:

        self.rescale_layer = Sequential(layers.Rescaling(1./255))
        self.augmentation  = Sequential(layers.RandomFlip("horizontal"))

        self.train_dir = train_dir
        self.test_dir  = test_dir
        

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


    def _load_data(self) -> None:
        
        print("Fetching datasets")

        train_ds = image_dataset_from_directory(self.train_dir,
                                                image_size = (28, 28),
                                                batch_size = 128,
                                                color_mode = "grayscale",
                                                subset = "training",
                                                seed = 1,
                                                validation_split = 0.2)

        train_ds = train_ds.map(lambda x, y: (self.rescale_layer(x), y), num_parallel_calls = AUTOTUNE)
        train_ds = train_ds.shuffle(len(train_ds)).prefetch(buffer_size = AUTOTUNE).cache()

        val_ds = image_dataset_from_directory(self.train_dir,
                                              image_size = (28, 28),
                                              batch_size = 128,
                                              color_mode = "grayscale",
                                              subset = "validation",
                                              seed = 1,
                                              validation_split = 0.2)

        val_ds = val_ds.map(lambda x, y: (self.rescale_layer(x), y), num_parallel_calls = AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size = AUTOTUNE).cache()

        test_ds = image_dataset_from_directory(self.test_dir,
                                               image_size = (28, 28),
                                               color_mode = "grayscale")

        # test_ds = test_ds.prefetch(buffer_size = AUTOTUNE).cache()

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

    
    def plot_sample(self, with_info: bool = True) -> None:
        pass

    
    def create_model(self) -> None:

        input_shape = (28, 28, 1)
        
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
        model.add(Dense(self.num_classes, activation = "softmax"))
        
        self.model = model


    def train_model(self) -> None:
        
        model = self.model
        model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

        self.model_history = model.fit(self.train_data,
                                       epochs = 10,
                                       verbose = 1,
                                       validation_data = self.val_data)

    def _get_trained_model(self) -> None:
        
        inputs = Input(shape = (28, 28, 1))
        x = self.rescale_layer(inputs)
        outputs = self.model(x)

        self.trained_model = Model(inputs, outputs)


    def evaluate_model(self) -> None:
        
        self._get_trained_model()
        
        loss, acc = self.trained_model.evaluate(self.test_data)


    def plot_confusion_matrix(self) -> None:
        pass


# %%

def main():

    train_dir = "C:/Users/Joao/Kotoamatsukami/Kotoamatsukami/data/fashion_mnist/train"
    test_dir  = "C:/Users/Joao/Kotoamatsukami/Kotoamatsukami/data/fashion_mnist/test"

    operator = CV(train_dir = train_dir, test_dir = test_dir)
    operator.get_data()
    operator.create_model()
    operator.train_model()
    operator.evaluate_model()
    
    predictions = model.predict(test_inputs)

    preds = [np.argmax(p) for p in predictions]

    confusion = confusion_matrix(labels = truth, predictions = preds, num_classes = 10)

    fig = px.imshow(confusion, color_continuous_scale = "viridis")
    fig.show()

if __name__ == "__main__":

    main()
# %%

# predictions = model.predict(operator.test_data)

preds = [np.argmax(p) for p in predictions]

confusion = confusion_matrix(labels = truth, predictions = preds, num_classes = 10)

fig = px.imshow(confusion, color_continuous_scale = "viridis")
fig.show()
# %%
