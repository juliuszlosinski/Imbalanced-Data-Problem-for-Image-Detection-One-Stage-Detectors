from balancers.balancer import Balancer

#methods=["SMOTE", "ADASYN", "AUGMENTATION", "DGAN", "AE"]
methods=["AE"]

print(methods)

#methods=["SMOTE", "ADASYN", "AUGMENTATION"]
balancer=Balancer("./balancer_configuration.json")
debug=True

print("Working 1")

balancer.print_configuration()
for method in methods:
    print("Working 2")
    path_to_input_imbalanced_images_folder=f"./maritime-flags-dataset/imbalanced_flags/images"                    # IMBALANCED/SRC IMAGES
    path_to_output_balanced_images_folder=f"./maritime-flags-dataset/{method}_balanced_flags/images/"             # BALANCED IMAGES
    path_to_output_balanced_images_annotations_folder=f"./maritime-flags-dataset/{method}_balanced_flags/labels/" # ANNOTATIONS
    balancer.fit(path_to_input_image_folder=path_to_input_imbalanced_images_folder, mode=method, debug=True)
    balancer.balance(path_to_output_image_folder=path_to_output_balanced_images_folder, debug=True)
    balancer.annotate(path_to_input_image_folder=path_to_output_balanced_images_folder, path_to_output_annotations=path_to_output_balanced_images_annotations_folder)
