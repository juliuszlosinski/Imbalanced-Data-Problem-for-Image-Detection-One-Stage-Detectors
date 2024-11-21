import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from image_util import ImageUtil

class ImageSMOTE:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
    
    """
    Fitting/loading the images and preprocessing to flatten vectors for SMOTE.
    """
    def fit(self, path_to_input_image_folder:str, width_of_image:int, height_of_image:int, debug:bool=False):
        self.images:list = []
        self.labels:list = []
        self.width_of_image = width_of_image
        self.height_of_image = height_of_image
        for directory_name in os.listdir(path_to_input_image_folder):
            if os.path.isdir(directory_name):
                print(True)
                for file_name in os.listidr(f"{path_to_input_image_folder}/{directory_name}"):
                    image = Image.open(f"{path_to_input_image_folder}/{file_name}")
                    image = image.resize((self.width_of_image, self.height_of_image))
                    image_array = np.array(image).flatten()
                    self.images.append(image_array)
                    self.labels.append(file_name.split('_')[0])  
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        if debug:
            print(f"images[0] = {self.images[0]}\n")
            print(f"labels[0] = {self.labels[0]}\n")
            print(f"X/images: \n{self.images}\n")
            print(f"y/labels: \n{self.labels}\n")

    """
    Counting the number of classes and plotting the histogram
    """
    def count(self, path_to_image_folder:str, show_image=False):
        files = os.listdir(path_to_image_folder)
        names = []
        for file_name in files:
            name = file_name.split("_")[0]
            names.append(name)
        names = set(names)
        data = {}
        for i in names:
            data[i]=0
        for file_name in files:
            name = file_name.split("_")[0]
            data[name] = int(data[name])+1
        print(data)
        x = [value[0] for value in data.items()]
        y = [value[1] for value in data.items()]
        if show_image:
            plt.bar(x, y)
            plt.grid(True)
            plt.xlabel("Classes")
            plt.ylabel("Count(classes)")
            plt.title("Count(classes) - Number of classes")
            plt.show()
        return data

    """
    Balancing the dataset.
    """
    def balance(self, path_to_output_image_folder:str, debug=False):
        labels_encoded = self.label_encoder.fit_transform(self.labels)
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        images_resampled, labels_resampled = smote.fit_resample(self.images, labels_encoded)
        if debug:
            print(f"Images resampled: \n{images_resampled}\n")
            print(f"Labels resampled: \n{labels_resampled}\n")
        if not os.path.exists(path_to_output_image_folder):
            os.makedirs(path_to_output_image_folder)
        for i in range(len(images_resampled)):
            image_array = images_resampled[i].reshape(self.width_of_image, self.height_of_image, 3)  
            image = Image.fromarray(np.uint8(image_array))
            label = self.label_encoder.inverse_transform([labels_resampled[i]])[0]
            file_name = f"{label}_{i}.png"
            image.save(f"{path_to_output_image_folder}/{file_name}")

#### CONFIGURATION #############
path_to_input_image_folder = "./dataset/traning/imbalanced_flags/"
path_to_output_image_folder = "./dataset/traning/balanced_flags"
width_of_image = 150
height_of_image = 150
debug = True
###############################

image_smote = ImageSMOTE()
#image_smote.fit(path_to_input_image_folder=path_to_input_image_folder, width_of_image=width_of_image, height_of_image=height_of_image, debug=debug)
image_smote.balance(path_to_output_image_folder=path_to_output_image_folder) # BALANCING THE DATASETs
#image_smote.count(path_to_image_folder=path_to_input_image_folder, show_image=debug)
#image_smote.count(path_to_image_folder=path_to_output_image_folder, show_image=debug)