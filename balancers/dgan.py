import shutil
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils

# Generator network for creating images
class Generator(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Generator, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dimension, 1024, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()  # Tanh to normalize to [-1, 1] for image pixel values
        )

    def forward(self, x):
        return self.layers(x)

# Discriminator network to distinguish real from generated images
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()  # Sigmoid to output probabilities
        )

    def forward(self, x):
        return self.layers(x).view(-1)

# GAN framework to handle training and prediction
class DGAN:
    def __init__(self, latent_dimension=100, learning_rate=0.002, beta_01=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dimension = latent_dimension
        self.generator_model = Generator(latent_dimension).to(self.device)
        self.discriminator_model = Discriminator().to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer_generator = torch.optim.Adam(self.generator_model.parameters(), lr=learning_rate, betas=(beta_01, 0.999))
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator_model.parameters(), lr=learning_rate, betas=(beta_01, 0.999))

    # Training method for GAN
    def train(self, dataset_path, batch_size=128, number_of_epochs=200, output_images_per_epoch=False):
        if output_images_per_epoch:
            output_path_training = f"./output_training_gan"
            if os.path.exists(output_path_training):
                shutil.rmtree(output_path_training)
            os.makedirs(output_path_training, exist_ok=True)

        # Data transformations and loader setup
        transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        # Check if dataset_path exists and contains images
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path {dataset_path} does not exist.")
            return
        
        print(dataset_path)
        
        class_name = str(dataset_path).split("/")[-1]
        tmp_path = f"{dataset_path}/{class_name}_tmp"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.mkdir(tmp_path)
        for file in os.listdir(dataset_path):
            if os.path.isfile(f"{dataset_path}/{file}"):
                shutil.copy2(f"{dataset_path}/{file}", tmp_path)
        
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        
        # Skip if no images are found
        if len(dataset) == 0:
            print(f"Warning: No images found in {dataset_path}. Skipping training for this directory.")
            return

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        fixed_noise = torch.randn(64, self.latent_dimension, 1, 1, device=self.device)

        for epoch in range(number_of_epochs):
            for i, (data, _) in enumerate(data_loader):
                real_data = data.to(self.device)
                batch_size = real_data.size(0)
                label_real = torch.ones(batch_size, device=self.device)
                label_fake = torch.zeros(batch_size, device=self.device)

                # Discriminator update
                self.discriminator_model.zero_grad()
                output_real = self.discriminator_model(real_data)
                loss_real = self.criterion(output_real, label_real)
                noise = torch.randn(batch_size, self.latent_dimension, 1, 1, device=self.device)
                fake_data = self.generator_model(noise)
                output_fake = self.discriminator_model(fake_data.detach())
                loss_fake = self.criterion(output_fake, label_fake)
                loss_discriminator = loss_real + loss_fake
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # Generator update
                self.generator_model.zero_grad()
                label_gen = torch.ones(batch_size, device=self.device)
                output_fake = self.discriminator_model(fake_data)
                loss_generator = self.criterion(output_fake, label_gen)
                loss_generator.backward()
                self.optimizer_generator.step()

                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{number_of_epochs}] Batch {i}/{len(data_loader)} Loss D: {loss_discriminator:.4f}, Loss G: {loss_generator:.4f}")

            # Save generated images for every epoch
            if output_images_per_epoch:
                with torch.no_grad():
                    fake_images = self.generator_model(fixed_noise).detach().cpu()
                    utils.save_image(fake_images, f"{output_path_training}/output_epoch_{epoch+1}.png", normalize=True)
                    print(f"Saved generated images for epoch {epoch+1}")
        shutil.rmtree(tmp_path)

    # Predict and save an image using the trained generator
    def predict(self, path_to_output_image):
        self.generator_model.eval()
        noise = torch.randn(1, self.latent_dimension, 1, 1, device=self.device)
        with torch.no_grad():
            fake_image = self.generator_model(noise).detach().cpu()
        utils.save_image(fake_image, path_to_output_image, normalize=True)

# Class to balance datasets by generating missing images for underrepresented categories
class DGANBalancer:
    def __init__(self):
        pass
    
    def fit(self, path_to_input_image_folder, latent_dimension=100, learning_rate=0.002, beta_01=0.5, batch_size=128, number_of_epochs= 200, delta=5):
        self.path_to_folder=path_to_input_image_folder        
        directories = os.listdir(path_to_input_image_folder)
        real_directories = {}
        to_generate = {}
        for directory in directories:
            if os.path.isdir(f"{path_to_input_image_folder}/{directory}"):
                count = len(os.listdir(f"{path_to_input_image_folder}/{directory}"))
                real_directories[directory]=count
        max_id = max(real_directories, key=real_directories.get)
        self.total_classes = [key for key in real_directories]
        for directory in real_directories:
            diff = real_directories[max_id] - real_directories[directory]
            if(diff>delta):
                to_generate[directory]=diff
        print(to_generate)
        self.maps = {}
        for key in to_generate:
            count = to_generate[key]
            self.maps[key]=(count, DGAN(
                latent_dimension=latent_dimension, 
                learning_rate=learning_rate, 
                beta_01=beta_01
            ))
        for key in self.maps:
            print(f"{key}: {self.maps[key][0]}, {self.maps[key][1]}")
            dgan_model = self.maps[key][1]
            dgan_model.train(
                dataset_path=f"{path_to_input_image_folder}/{key}",
                batch_size=batch_size,
                number_of_epochs=number_of_epochs
            )
            
    def balance(self, path_to_output_image_folder, debug=False):
        for category in self.total_classes:
            source_folder = f"{self.path_to_folder}/{category}"
            destination_folder =f"{path_to_output_image_folder}/{category}"
            if os.path.exists(destination_folder) and os.path.isdir(destination_folder):
                shutil.rmtree(destination_folder)
            shutil.copytree(source_folder, destination_folder)
        if debug:
            print(self.maps)
        for category in self.maps:
            last_id = -1
            for file in os.listdir(f"{path_to_output_image_folder}/{category}"):
                id = int(file.split("_")[1].split(".")[0])
                if id > last_id:
                    last_id = id
            for i in range(self.maps[category][0]):
                self.maps[category][1].predict(f"{path_to_output_image_folder}/{category}/{category}_{last_id}.jpg")
                last_id = last_id + 1