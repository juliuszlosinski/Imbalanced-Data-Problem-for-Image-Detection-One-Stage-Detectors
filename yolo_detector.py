import os
import shutil
import random
from ultralytics import YOLO
import yaml

class YOLODetector:
    def __init__(self, type="yolov8s.pt"):
        self.yolo_type = type
        self.model = YOLO(self.yolo_type)
    
    def create_train_val_test_sets(self, path_to_source_directory, path_to_output_directory, train_percent, test_percent=0.2, val_percent=0.1, shuffle=True):
        if not (0 <= train_percent <= 1 and 0 <= test_percent <= 1 and 0 <= val_percent <= 1):
            raise ValueError("Percentages must be between 0 and 1.")
        if not abs(train_percent + test_percent + val_percent - 1) < 1e-6:
            raise ValueError("Percentages must sum to 1.")
        # 1. Removing all folders/ files from output directory.
        if not os.path.exists(path_to_output_directory):
            print(f"Directory {path_to_output_directory} does not exist.")
            return
        for item in os.listdir(path_to_output_directory):
            item_path = os.path.join(path_to_output_directory, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path) 
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
                
        # 2. Creating train, val and test directories.
        os.makedirs(f"{path_to_output_directory}/images")
        os.makedirs(f"{path_to_output_directory}/images/train")
        os.makedirs(f"{path_to_output_directory}/images/val")
        os.makedirs(f"{path_to_output_directory}/images/test")
        
        os.makedirs(f"{path_to_output_directory}/labels")
        os.makedirs(f"{path_to_output_directory}/labels/train")
        os.makedirs(f"{path_to_output_directory}/labels/val")
        os.makedirs(f"{path_to_output_directory}/labels/test")
        
        # 3. Saving path to images and labels from directory.
        path_to_source_images = f"{path_to_source_directory}/images"
        target_train_images = f"{path_to_output_directory}/images/train"
        target_val_images = f"{path_to_output_directory}/images/val"
        target_test_images = f"{path_to_output_directory}/images/test"
        
        path_to_source_labels = f"{path_to_source_directory}/labels"
        target_train_labels = f"{path_to_output_directory}/labels/train"
        target_val_labels = f"{path_to_output_directory}/labels/val"
        target_test_labels = f"{path_to_output_directory}/labels/test"
        
        # 4. Saving all paths to training images.
        paths_to_images = []
        path_to_image = path_to_source_images
        for category in os.listdir(path_to_image):
            path_to_category = f"{path_to_image}/{category}"
            for image in os.listdir(path_to_category):
                images = os.listdir(path_to_category)
                paths_to_images.append(f"{path_to_category}/{image}")
        if shuffle:
            random.shuffle(paths_to_images)
        n = len(paths_to_images)
        train_end = int(train_percent * n)
        test_end = train_end + int(test_percent * n)
        train_set = paths_to_images[:train_end]
        test_set = paths_to_images[train_end:test_end]
        val_set = paths_to_images[test_end:]
        
        print(f"Length of train set: {len(train_set)}")
        print(f"Length of val set: {len(val_set)}")
        print(f"Length of test set: {len(test_set)}")
        
        # 5. Moving images to specific categories (training, val, test).
        for source_image_path in train_set:
            source_image_file = str(source_image_path).split("/")[-1]
            source_label_file = str(source_image_file).split(".")[0]
            source_label_file = f"{source_label_file}.txt"
            category = source_label_file.split("_")[0]
            target_image_path = f"{target_train_images}/{source_image_file}"
            target_label_path = f"{target_train_labels}/{source_label_file}"
            source_label_path = f"{path_to_source_labels}/{category}/{source_label_file}"
            shutil.copy(source_image_path, target_image_path)
            shutil.copy(source_label_path, target_label_path)
            
        for source_image_path in val_set:
            source_image_file = str(source_image_path).split("/")[-1]
            source_label_file = str(source_image_file).split(".")[0]
            source_label_file = f"{source_label_file}.txt"
            category = source_label_file.split("_")[0]
            target_image_path = f"{target_val_images}/{source_image_file}"
            target_label_path = f"{target_val_labels}/{source_label_file}"
            source_label_path = f"{path_to_source_labels}/{category}/{source_label_file}"
            shutil.copy(source_image_path, target_image_path)
            shutil.copy(source_label_path, target_label_path)
            
        for source_image_path in test_set:
            source_image_file = str(source_image_path).split("/")[-1]
            source_label_file = str(source_image_file).split(".")[0]
            source_label_file = f"{source_label_file}.txt"
            category = source_label_file.split("_")[0]
            target_image_path = f"{target_test_images}/{source_image_file}"
            target_label_path = f"{target_test_labels}/{source_label_file}"
            source_label_path = f"{path_to_source_labels}/{category}/{source_label_file}"
            shutil.copy(source_image_path, target_image_path)
            shutil.copy(source_label_path, target_label_path)
    
    def create_cross_validation_set(self, path_to_source_directory, path_to_output_directory, number_of_k_folds=5, shuffle=True):
        path_to_source_labels = f"{path_to_source_directory}/labels"
        path_to_source_images = f"{path_to_source_directory}/images"
        class_names = os.listdir(f"{path_to_source_directory}/images")
        self.classes = class_names
        self.number_of_folds = number_of_k_folds
        paths_to_images = []
        for category in os.listdir(path_to_source_images):
            category_path = f"{path_to_source_images}/{category}"
            if os.path.isdir(category_path):
                for image in os.listdir(category_path):
                    image_path = f"{category_path}/{image}"
                    paths_to_images.append(image_path)
        
        if shuffle:
            random.shuffle(paths_to_images)
        
        n = len(paths_to_images)
        fold_sizes = [n // number_of_k_folds] * number_of_k_folds
        for i in range(n % number_of_k_folds):
            fold_sizes[i] += 1

        paths_to_images_folds = []
        start = 0
        for size in fold_sizes:
            paths_to_images_folds.append(paths_to_images[start:start + size])
            start += size
        
        if os.path.isdir(path_to_output_directory):
            shutil.rmtree(path_to_output_directory)
        os.makedirs(path_to_output_directory)
        os.makedirs(f"{path_to_output_directory}/images")
        os.makedirs(f"{path_to_output_directory}/labels")
        
        for i, fold in enumerate(paths_to_images_folds):
            fold_dir = f"{path_to_output_directory}/images/fold_{i+1}"
            label_dir = f"{path_to_output_directory}/labels/fold_{i+1}"
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            num_train = int(len(fold) * 0.8)  
            train_set = fold[:num_train]
            val_set = fold[num_train:]

            os.makedirs(f"{fold_dir}/train", exist_ok=True)
            os.makedirs(f"{fold_dir}/val", exist_ok=True)
            os.makedirs(f"{label_dir}/train", exist_ok=True)
            os.makedirs(f"{label_dir}/val", exist_ok=True)

            for image_path in train_set:
                image_file = os.path.basename(image_path)
                label_file = os.path.splitext(image_file)[0] + ".txt"
                label_category = label_file.split("_")[0]  

                label_source_path = f"{path_to_source_labels}/{label_category}/{label_file}"
                shutil.copy(image_path, f"{fold_dir}/train/{image_file}")
                shutil.copy(label_source_path, f"{label_dir}/train/{label_file}")

            for image_path in val_set:
                image_file = os.path.basename(image_path)
                label_file = os.path.splitext(image_file)[0] + ".txt"
                label_category = label_file.split("_")[0]
                label_source_path = f"{path_to_source_labels}/{label_category}/{label_file}"
                shutil.copy(image_path, f"{fold_dir}/val/{image_file}")
                shutil.copy(label_source_path, f"{label_dir}/val/{label_file}")
        print("Cross-validation set created successfully.")
       
    def evaluate_cross_validation(self, name_of_dataset, image_size=128, number_of_epochs=32):
        for k in range(self.number_of_folds):
            train_dir = f"./{name_of_dataset}/images/fold_{k+1}/train"
            val_dir = f"./{name_of_dataset}/images/fold_{k+1}/val"
            yaml_file_path = f"./fold_{k+1}_dataset.yaml"
            
            # Create the dataset.yaml for training
            yaml_data = {
                'train': train_dir,
                'val': val_dir,
                'nc': len(self.classes),
                'names': self.classes
            }
            
            with open(yaml_file_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)

            # Train the model for the fold
            results = self.model.train(
                data=yaml_file_path,
                imgsz=image_size,
                epochs=number_of_epochs
            )
            
            print(f"Results: {results}")
            precision = results.results_dict.get('precision', 0)
            recall = results.results_dict.get('recall', 0)
            f1_score = results.results_dict.get('f1_score', 0)
            print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")
        
    def fit(self, configuration_file, image_size=128, number_of_epochs=32):
        self.model.train(data=configuration_file, imgsz=image_size, epochs=number_of_epochs)
        
    def predict(self, path_to_image=128):
        self.model(path_to_image, save=True, save_txt=True)