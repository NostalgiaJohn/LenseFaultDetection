import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


class Data:
    def __init__(self):
        # Load your image and mask
        image = Image.open("./train/image.bmp").convert("L") # Convert mask to grayscale
        mask = Image.open("./train/mask.bmp").convert("L")

        # Convert image and mask to tensors
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        # Set patch size
        patch_size = (50, 50)

        # Calculate number of patches along each dimension
        rows = image.size(1) // patch_size[0]
        cols = image.size(2) // patch_size[1]

        # Lists to store patches and labels
        image_patches = []
        labels = []

        # Slice the image and mask into patches and determine labels
        for i in range(rows):
            for j in range(cols):
                # Extract patch from image and mask
                img_patch = image[:, i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]]
                mask_patch = mask[:, i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]]

                # Determine if the patch is in the area of interest
                is_interest = mask_patch.mean().item() > 0.  # Threshold: more than 50% of pixels are interest

                # Append patch and label
                image_patches.append(img_patch)
                labels.append(is_interest)

        # Convert lists to tensors
        image_patches = torch.stack(image_patches)
        labels = torch.tensor(labels)

        # Create dataset and dataloader
        dataset = TensorDataset(image_patches, labels.unsqueeze(1).float())  # Reshape labels for BCE Loss
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
