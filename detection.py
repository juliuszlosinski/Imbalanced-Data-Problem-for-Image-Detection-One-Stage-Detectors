from yolo_detector import YOLODetector

#############################
shuffle = True
image_size = 128
number_of_epochs = 5
path_to_source_directory = "./maritime-flags-dataset/balanced_two_flags/" # TO CHANGE! -> A and B
project_name = "yolo-results"
experiment_name = f"Balanced-Two-Flags"

# CROSS VALIDATION
cross_validation_target_path = "./datasets/cross-validation-yolo-formatted-maritime-flags/"
name_of_dataset = "cross-validation-yolo-formatted-maritime-flags" # Name of directory in datasets folder
number_of_k_folds = 5

# DEFAULT TRAIN/VAL/TEST split
default_target_path = "./datasets/yolo-formatted-maritime-flags/"
default_configuration_file = "./yolo_data.yaml" # TO CHANGE! -> A and B
train_percent = 0.7 # 70 [%]
test_percent = 0.2  # 20 [%]
val_percent = 0.1   # 10 [%]

yolo_type = "yolo11m.pt"
# yolo_type = "yolov8m.pt"
detector = YOLODetector(type=yolo_type) # M ~ medium version, S ~ small version.

mode = 0 # 0: Train/Val/Test mode, 1: Cross validation mode
if mode:
    experiment_name = f"Cross_validation-{experiment_name}"
    detector.create_cross_validation_set(
        path_to_source_directory = path_to_source_directory,
        path_to_output_directory = cross_validation_target_path,
        number_of_k_folds = number_of_k_folds,
        shuffle = shuffle
    )
    detector.evaluate_cross_validation(
        name_of_dataset="cross-validation-yolo-formatted-maritime-flags",
        image_size = image_size,
        number_of_epochs = number_of_epochs,
        project_name=project_name,
        experiment_name=experiment_name
    )
else:
    experiment_name = f"{experiment_name}-Epochs_{number_of_epochs}"
    detector.create_train_val_test_sets(
        path_to_source_directory=path_to_source_directory, 
        path_to_output_directory=default_target_path, 
        train_percent=train_percent, 
        test_percent=test_percent, 
        val_percent=val_percent, 
        shuffle = shuffle
    )
    detector.fit(
        configuration_file=default_configuration_file, 
        image_size=image_size,
        number_of_epochs=number_of_epochs,
        project_name=project_name,
        experiment_name=experiment_name
    )