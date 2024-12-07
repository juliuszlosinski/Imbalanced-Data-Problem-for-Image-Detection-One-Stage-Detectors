from yolo_formatter import YOLOFormatter
from ultralytics import YOLO
import random
import shutil

class YOLODetector:
    def __init__(self, yolo_type="yolov8s.pt"):
        self.yolo_formatter = YOLOFormatter()
        self.yolo_type = yolo_type
        self.model = YOLO(yolo_type)
    
    def format_to_yolo_directories(self, images_source_directory, labels_source_directory):
        images_target_directory = "./datasets/yolo-maritime-flags-dataset/images/train/"
        labels_target_directory = "./datasets/yolo-maritime-flags-dataset/labels/train/"
        self.yolo_formatter.delete_all_files_and_directories(images_target_directory)
        self.yolo_formatter.delete_all_files_and_directories(labels_target_directory)
        self.yolo_formatter.copy(images_source_directory, images_target_directory)
        self.yolo_formatter.copy(labels_source_directory, labels_target_directory)
    
    def create_validation_set(self, images_source_directory, labels_source_directory, number_of_images_per_class):
        images_target_directory = "./datasets/yolo-maritime-flags-dataset/images/val/"
        labels_target_directory = "./datasets/yolo-maritime-flags-dataset/labels/val/"
        if os.path.exists(images_target_directory):
            self.yolo_formatter.delete_all_files_and_directories(images_target_directory)
        if os.path.exists(labels_target_directory):
            self.yolo_formatter.delete_all_files_and_directories(labels_target_directory)
        os.makedirs(images_target_directory)
        os.makedirs(labels_target_directory)
        categories = os.listdir(images_source_directory)
        for category in categories:
            path_to_category = f"{images_source_directory}/{category}"
            number_of_images = len(os.listdir(path_to_category)) # 400
            lower_bound = 10
            upper_bound = number_of_images
            selected_images = [f"{category}_{random.randint(lower_bound, upper_bound)}.jpg" for _ in range(number_of_images_per_class)]
            for image in selected_images:
                source=f"{path_to_category}/{image}"
                destination=f"{images_target_directory}/{image}"
                shutil.copy(source, destination)
                label=str(image).split(".")[0]+".txt"
                path_to_source_label=f"{labels_source_directory}/{category}/{label}"
                path_to_output_label=f"{labels_target_directory}/{label}"
                shutil.copy(path_to_source_label, path_to_output_label)

    def create_test_set(self,images_source_directory, labels_source_directory, number_of_images_per_class):
        images_target_directory = "./datasets/yolo-maritime-flags-dataset/images/test/"
        labels_target_directory = "./datasets/yolo-maritime-flags-dataset/labels/test/"
        if os.path.exists(images_target_directory):
            self.yolo_formatter.delete_all_files_and_directories(images_target_directory)
        if os.path.exists(labels_target_directory):
            self.yolo_formatter.delete_all_files_and_directories(labels_target_directory)
        os.makedirs(images_target_directory)
        os.makedirs(labels_target_directory)
        categories = os.listdir(images_source_directory)
        for category in categories:
            path_to_category = f"{images_source_directory}/{category}"
            number_of_images = len(os.listdir(path_to_category)) # 400
            lower_bound = 10
            upper_bound = number_of_images
            selected_images = [f"{category}_{random.randint(lower_bound, upper_bound)}.jpg" for _ in range(number_of_images_per_class)]
            for image in selected_images:
                source=f"{path_to_category}/{image}"
                destination=f"{images_target_directory}/{image}"
                shutil.copy(source, destination)
                label=str(image).split(".")[0]+".txt"
                path_to_source_label=f"{labels_source_directory}/{category}/{label}"
                path_to_output_label=f"{labels_target_directory}/{label}"
                shutil.copy(path_to_source_label, path_to_output_label)
    
    def fit(self, configuration_file, image_size, number_of_epochs):
        self.model.train(data=configuration_file, imgsz=image_size, epochs=number_of_epochs)
        
    def predict(self, path_to_image):
        self.model(path_to_image, save=True, save_txt=True)