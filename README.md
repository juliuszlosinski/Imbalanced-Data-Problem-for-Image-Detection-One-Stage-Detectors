## 1. UML
![image](https://github.com/user-attachments/assets/91ec7ad1-c5ac-4c0a-a183-0dbfcab81b1f)

## 2. Project Organization
```
├── documentation       <- UML diagrams
├── balancers           <- Package with balancers and utils
│   ├── __init__.py     <- Package identicator
│   ├── smote.py        <- SMOTE balancer (interpolation)
│   ├── adasyn.py       <- ADASYN balancer (interpolation)
│   ├── augmentation.py <- Augmentation balancer (augmenting images like rotations, etc.)
│   ├── autoencoder.py  <- Autoencoder balancer (learning needed!)
│   ├── dgan.py         <- DGAN balancer (learning needed!)
│   ├── balancer.py     <- General balancer with all balancers (aggregating all of the above)
│   ├── annotations.py  <- Annotations module
│   └── configuration_reader.py  <- Balancer configuration reader
├── maritime-flags-dataset    <- Source and balanced flags (A-Z)
│   ├── ADASYN_balanced_flags <- Balanced flags by using ADASYN balancer
│   ├── SMOTE_balanced_flags  <- Balanced flags by using SMOTE balancer
│   ├── AUGMENTATION_balanced_flags  <- Balanced flags by using Augmentation balancer
│   ├── DGAN_balanced_flags  <- Balanced flags by using DGAN balancer
│   ├── AE_balanced_flags    <- Balanced flags by using Autoencoder balancer
│   ├── combined_flags       <- Combined/test images 
│   ├── two_flags            <- Balanced two flags (A and B) per 1000 images
│   └── imbalanced_flags     <- Source folder with imbalanced flags
├── datasets   <- YOLO formatted datasets (detector by default sees this category!)
│   ├── yolo-maritime-flags-dataset (A-Z)
│     ├── images
│       ├── train <- Training images (.jpg)
│       ├── val   <- Validation images (.jpg)
│       └── test  <- Testing images (.jpg)
│     └── labels
│       ├── train <- Training labels (.txt)
│       ├── val   <- Validation labels (.txt)
│       └── test  <- Testing labels (.txt)
│   └── cross-validation-yolo-formatted-maritime-flags (A-Z)
│     ├── images
│       ├── fold_1  <- First fold with images (.jpg)
|         ├── train <- Training images (.jpg)
|         └── val   <- Validation images (.jpg)
│       ├── ...     <- ... fold with images (.jpg)
|         ├── train <- Training images (.jpg)
|         └── val   <- Validation images (.jpg)
│       └── fold_n  <- N-fold with images (.jpg)
|         ├── train <- Training images (.jpg)
|         └── val   <- Validation images (.jpg)
│     └── labels
│       ├── fold_1  <- First fold with labels (.txt)
|         ├── train <- Training labels (.txt)
|         └── val   <- Validation labels (.txt)
│       ├── ...     <- ... fold with labels (.txt)
|         ├── train <- Training labels (.txt)
|         └── val   <- Validation labels (.txt)
│       └── fold_n  <- N-fold with labels (.txt)
|         ├── train <- Training labels (.txt)
|         └── val   <- Validation labels (.txt)
├── balance.py <- Balancing dataset by using balancers package (BALANCING)
├── balancer_configuration.json <- Balancer configuration
├── detection.py     <- Training and testing yolo detector with balanced/imbalanced data (EVALUATING)
├── yolo_data.yaml   <- YOLO data configuration
└── yolo_detector.py <- YOLO detector (DETECTING)
```

## 3. Balancing approaches

### 3.1. Augmentation
![image](https://github.com/user-attachments/assets/d97ffb0f-f56e-499f-a6a3-c141c7b9d27c)
![image](https://github.com/user-attachments/assets/ab7e208d-e907-4bf0-bcef-4ce9a17d9e74)

### 3.2. SMOTE
![image](https://github.com/user-attachments/assets/4ae01470-abc5-45e3-a9cb-3f26f25d7564)
![image](https://github.com/user-attachments/assets/07576e54-dd8e-4abf-b47a-6e9fcb605eb4)
![image](https://github.com/user-attachments/assets/3491bfce-3665-4768-b996-18ceb3701a62)

### 3.3. ADASYN
![image](https://github.com/user-attachments/assets/7585f729-a99f-4d8a-9101-82c781084a5b)
![image](https://github.com/user-attachments/assets/5f49fd46-6bf3-4dc9-ab10-b8c242cbe11b)

### 3.4. Autoencoder
![image](https://github.com/user-attachments/assets/d7c0f3cf-1b1d-4e40-ba8b-7067285a9b03)
![image](https://github.com/user-attachments/assets/a258c8e6-738a-4f49-83a9-551c0e417edd)
![image](https://github.com/user-attachments/assets/e1b2da08-eef8-42b9-8601-4f2be1fe45a7)
![image](https://github.com/user-attachments/assets/fa11a4cf-65f0-4c4a-8154-30b7e946c234)
![image](https://github.com/user-attachments/assets/983a0fb4-e56b-4d4d-8cfa-05634f8ae479)

## 3.5. Deep Convolutional GAN
![image](https://github.com/user-attachments/assets/627426ed-7030-46e5-b0e3-4a4a6dfb9237)
![image](https://github.com/user-attachments/assets/eb4c63a1-5853-4534-adba-d06291a44dfd)
