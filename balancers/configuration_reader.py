import json

class ConfigurationReader():
    def __init__(self):
        self.initialized = False
        
    def read(self, path_to_configuration_file):
        self.initialized = True
        json_file = open(path_to_configuration_file)
        conf = json.loads(json_file.read())
        self.path_to_input_imbalanced_images_folder=conf["path_to_input_imbalanced_images_folder"]
        self.path_to_output_balanced_images_folder=conf["path_to_output_balanced_images_folder"]
        self.width_of_image=conf["width_of_image"]
        self.height_of_image=conf["height_of_image"]
        self.number_of_channels=conf["number_of_channels"]
        self.batch_size=conf["batch_size"]
        self.number_of_epochs=conf["number_of_epochs"]
        self.delta=conf["delta"]
        self.number_of_neighbors=conf["number_of_neighbors"]
        self.latent_dimension=conf["latent_dimension"]
        self.learning_rate=conf["learning_rate"]
        self.beta=conf["beta"]
        json_file.close()
        
    def print(self):
        if self.initialized is True:
            print(f"Path to input imbalanced images folder: {self.path_to_input_imbalanced_images_folder}\n")
            print(f"Path to output balanced images folder: {self.path_to_output_balanced_images_folder}\n")
            print(f"Size of image: ({self.width_of_image}, {self.height_of_image}, {self.number_of_channels})\n")
            print(f"Batch size: {self.batch_size}\n")
            print(f"Number of epochs: {self.number_of_epochs}\n")
            print(f"Latent dimension: {self.latent_dimension}\n")
            print(f"Learning rate: {self.learning_rate}\n")
            print(f"Beta: {self.beta}")
        