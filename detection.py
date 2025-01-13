from yolo_detector import YOLODetector

# Configuration
shuffle = True
image_size = 128
number_of_epochs = 100
project_name = "yolo-results"
default_target_path = "./datasets/yolo-formatted-maritime-flags/"
default_configuration_file = "./yolo_data.yaml"
train_percent = 0.7  # 70 [%]
test_percent = 0.2   # 20 [%]
val_percent = 0.1    # 10 [%]
balance_datasets = [
    "imbalanced_flags", "SMOTE_balanced_flags", "ADASYN_balanced_flags",
    "AUGMENTATION_balanced_flags", "DGAN_balanced_flags", "AE_balanced_flags"
]
yolo_types = ["yolo11m.pt", "yolov8m.pt"]

# Training Loop
for yolo_type in yolo_types:
    detector = YOLODetector(type=yolo_type) 
    for balance_method in balance_datasets:
        try:
            path_to_source_directory = f"./maritime-flags-dataset/{balance_method}/"
            experiment_name = f"{yolo_type}_{balance_method}"
            print(f"Currently: {experiment_name}")
            
            # Create train/val/test sets
            print(f"Preparing dataset for {experiment_name}...")
            detector.create_train_val_test_sets(
                path_to_source_directory=path_to_source_directory, 
                path_to_output_directory=default_target_path, 
                train_percent=train_percent, 
                test_percent=test_percent, 
                val_percent=val_percent, 
                shuffle=shuffle
            )
            
            # Train the YOLO model
            print(f"Training {experiment_name}...")
            detector.fit(
                configuration_file=default_configuration_file, 
                image_size=image_size,
                number_of_epochs=number_of_epochs,
                project_name=project_name,
                experiment_name=experiment_name
            )
        except Exception as e:
            print(f"Error with {experiment_name}: {e}")