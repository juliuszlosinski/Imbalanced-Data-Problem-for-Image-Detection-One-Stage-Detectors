import os
import shutil

class YOLOFormatter:
    def __init__(self):
        pass
    
    def delete_all_files_and_directories(self, path_to_directory):
        if os.path.exists(path_to_directory):
            for root, dirs, files in os.walk(path_to_directory, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path_to_directory)
        else:
            print(f"The path {path} does not exist.")
    
    def copy(self, source_directory, target_directory):
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_directory, file)
                shutil.copy2(source_file, target_file)