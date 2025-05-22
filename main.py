from dataset import Data
from Network import Network

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


class Process:
    def __init__(self):
        # Load data
        self.data = Data()
        self.dataloader = self.data.dataloader

        # Instantiate model, define loss and optimizer
        self.model = Network().to("cuda")
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def Train(self):
        # Training loop
        num_epochs = 100

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.dataloader:
                # Forward pass
                outputs = self.model(inputs.to("cuda"))
                loss = self.criterion(outputs, labels.to("cuda"))

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update weights

                running_loss += loss.item()

            # Print epoch loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.dataloader):.4f}")
            # self.Test()

    def Test(self, threshold=0.5):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradient calculation
            # Convert the image to a tensor
            img_tensor = self.data.image_T_list[6]
            # img_tensor = self.add_gaussian_noise(self.data.image)
            # img_tensor = self.add_salt_and_pepper_noise(self.data.image)
            patch_size = self.data.patch_size

            img_height, img_width = img_tensor.shape[1:]

            # Calculate number of patches along each dimension
            rows = img_height // patch_size[0]
            cols = img_width // patch_size[1]

            # Initialize a blank mask for overlaying interest regions
            interest_mask = torch.zeros(1, img_height, img_width)
            line_width = 3
            # Loop through patches to build the full interest_mask based on model predictions
            for i in range(rows):
                for j in range(cols):
                    # Extract patch
                    patch = img_tensor[:, i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]]

                    # Run patch through the model
                    prediction = self.model(patch.unsqueeze(0).to("cuda")).item()

                    # Mark patch as "interest" if above threshold
                    if prediction >= threshold:
                        interest_mask[:,
                                      (i*patch_size[0]):((i + 1)*patch_size[0]),
                                      (j*patch_size[1]):((j + 1)*patch_size[1])] = 1
                        interest_mask[:,
                                      (i*patch_size[0] + line_width):((i + 1)*patch_size[0] - line_width),
                                      (j*patch_size[1] + line_width):((j + 1)*patch_size[1] - line_width)] = 0

            # Check interest_mask visually (for debugging)
            # plt.figure()
            # plt.title("Interest Mask Debug View")
            # plt.imshow(interest_mask.squeeze(0), cmap="gray")
            # plt.axis("off")
            # plt.show()

            self.visualize(img_tensor, interest_mask)

    def visualize(self, image, mask):
        # Remove single-channel dimension and convert to numpy array
        image = image.squeeze(0).cpu().numpy()
        mask = mask.squeeze(0).cpu().numpy()

        # Scale grayscale image if needed and convert to uint8
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)

        # Convert grayscale image to RGB
        rgb_image = np.stack([image] * 3, axis=-1)  # Convert to RGB

        # Overlay red where interest_mask is 1
        rgb_image[mask == 1] = [255, 0, 0]  # Set red pixels for regions of interest
        # Convert back to a PIL Image
        overlay_image = Image.fromarray(rgb_image)

        # Display the image pixel-to-pixel
        # overlay_image.show()
        # Display the image in PyCharm plot window
        plt.figure(figsize=(rgb_image.shape[1] / 100, rgb_image.shape[0] / 100), dpi=100)
        plt.imshow(overlay_image)
        plt.axis("off")
        plt.show()

    def add_gaussian_noise(self, image, mean=0, std=25):
        # Convert the image to a numpy array
        image_np = np.array(image).astype(np.float32)

        # Generate Gaussian noise
        noise = np.random.normal(mean, std, image_np.shape)

        # Add the noise to the image and clip values to stay within valid range
        noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL image
        noisy_image_T = T.ToTensor()(Image.fromarray(noisy_image))
        return noisy_image_T

    def add_salt_and_pepper_noise(self, image, prob=0.05):
        """
        Add salt and pepper noise to an image.

        Parameters:
        - image: PIL Image object, the input image.
        - prob: float, probability of each pixel being affected by noise (default: 0.05).

        Returns:
        - noisy_image_pil: PIL Image with salt and pepper noise.
        """
        # Convert image to numpy array
        image_np = np.array(image)

        # Generate random noise mask
        noisy_image = image_np.copy()
        num_salt = int(prob * image_np.size * 0.5)
        num_pepper = int(prob * image_np.size * 0.5)

        # Add salt (white) noise
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image_np.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        # Add pepper (black) noise
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image_np.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        # Convert back to PIL image
        noisy_image_T = T.ToTensor()(Image.fromarray(noisy_image))
        return noisy_image_T

    def save_model(self, file_path="model.pth"):
        """Save the model's state dictionary and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="model.pth"):
        """Load the model's state dictionary and optimizer state."""
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {file_path}")
        else:
            print(f"No checkpoint found at {file_path}")

if __name__ == "__main__":
    instance = Process()
    # instance.Train()
    # instance.save_model()
    instance.load_model()
    # for _ in range(20):
    instance.Test()
