
from typing import Dict, List
import os

from tensorflow.data import Dataset
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, Rescaling, RandomFlip
from tensorflow.math import confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
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
    
    
    def __init__(self, train_dir: str, test_dir: str, dataset, labels_dict: Dict[int, str] = None) -> None:

        self.train_dir = train_dir
        self.test_dir  = test_dir

        self.rescale_layer = Sequential(Rescaling(1./255))
        self.augmentation  = Sequential(RandomFlip("horizontal"))

        self.labels_dict = labels_dict
        self.dataset     = dataset

        self.get_data()
        self._infer_input_shape()


    def _donwload_data(self) -> None:
        
        print("Downloading datasets")

        data = self.dataset.load_data()

        train_data, test_data = data

        self.num_classes = len(np.unique([label for label in test_data[1]]))
        self.train_data  = train_data
        self.test_data   = test_data
    
    
    def _save_data(self) -> None:

        print("Saving the dataset to disk")

        # labels = {0: "T-shirt",
        #           1: "Trouser",
        #           2: "Pullover",
        #           3: "Dress",
        #           4: "Coat",
        #           5: "Sandal",
        #           6: "Shirt",
        #           7: "Sneaker",
        #           8: "Bag",
        #           9: "Ankle boot"}

        labels = self.labels_dict
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

        self.classes_names = os.listdir(train_dir)

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
            self.num_classes = len(self.classes_names)