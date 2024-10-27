from dataset import Data
from Network import Network

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from PIL import Image


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
            img_tensor = self.data.image_T.unsqueeze(0)
            patch_size = self.data.patch_size

            img_height, img_width = img_tensor.shape[2:]

            # Calculate number of patches along each dimension
            rows = img_height // patch_size[0]
            cols = img_width // patch_size[1]

            # Initialize a blank mask for overlaying interest regions
            interest_mask = torch.zeros(1, img_height, img_width)
            # Loop through patches to build the full interest_mask based on model predictions
            for i in range(rows):
                for j in range(cols):
                    # Extract patch
                    patch = img_tensor[:, :, i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]]

                    # Run patch through the model
                    prediction = self.model(patch.to("cuda")).item()

                    # Mark patch as "interest" if above threshold
                    if prediction >= 0.5:
                        interest_mask[:, i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = 1

            # Check interest_mask visually (for debugging)
            plt.figure()
            plt.title("Interest Mask Debug View")
            plt.imshow(interest_mask.squeeze(0), cmap="gray")
            plt.axis("off")
            plt.show()

            # Convert the full interest_mask to a PIL image for overlay
            interest_overlay = F.to_pil_image(interest_mask.squeeze(0) * 255).convert("L")

            # Define an RGBA version of the interest overlay with a solid color
            overlay_color = (255, 0, 0, 180)  # Red color with stronger opacity
            interest_overlay_rgba = Image.new("RGBA", interest_overlay.size, overlay_color)
            interest_overlay_rgba.putalpha(interest_overlay)  # Apply the mask as the alpha channel

            # Convert the original image to RGBA mode
            image_with_overlay = self.data.image.convert("RGBA")

            # Paste the overlay onto the original image once
            image_with_overlay.paste(interest_overlay_rgba, (0, 0), interest_overlay_rgba)

            # # Plot the original and overlaid image
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.title("Original Image")
            # plt.imshow(self.data.image)
            # plt.axis("off")
            #
            # plt.subplot(1, 2, 2)
            # plt.title("Interest Regions Overlay")
            # plt.imshow(image_with_overlay)
            # plt.axis("off")
            #
            # plt.show()



if __name__ == "__main__":
    process = Process()
    process.Train()
    process.Test()
