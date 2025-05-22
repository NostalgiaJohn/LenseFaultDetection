import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class Data:
    def __init__(self):
        # Load your image and mask
        image = Image.open("../../Downloads/LenseFaultDetection-master/train/image.bmp").convert("L") # Convert mask to grayscale
        mask = Image.open("../../Downloads/LenseFaultDetection-master/train/mask0.bmp").convert("L")
        image_new = Image.open("../../Downloads/LenseFaultDetection-master/train/image_new.bmp").convert("L") # Convert mask to grayscale
        mask_new = Image.open("../../Downloads/LenseFaultDetection-master/train/mask_new.bmp").convert("L")
        # Convert image and mask to tensors
        image_T = T.ToTensor()(image)
        mask_T = T.ToTensor()(mask)
        image_T_new = T.ToTensor()(image_new)
        mask_T_new = T.ToTensor()(mask_new)

        # Set patch size
        patch_size = (25, 25)
        # Calculate number of patches along each dimension
        rows = image_T.size(1) // patch_size[0]
        cols = image_T.size(2) // patch_size[1]

        # Prepare rotated images and masks
        image_T_R1 = F.rotate(image_T, 90)
        image_T_R2 = F.rotate(image_T, 180)
        image_T_R3 = F.rotate(image_T, 270)
        image_T_gauss = self.add_gaussian_noise(image)
        image_T_spn   = self.add_salt_and_pepper_noise(image)
        mask_T_R1 = F.rotate(mask_T, 90)
        mask_T_R2 = F.rotate(mask_T, 180)
        mask_T_R3 = F.rotate(mask_T, 270)
        # Lists to store patches and labels
        image_T_list = [image_T, image_T_R1, image_T_R2, image_T_R3, image_T_gauss, image_T_spn, image_T_new]
        mask_T_list = [mask_T, mask_T_R1, mask_T_R2, mask_T_R3, mask_T, mask_T, mask_T_new]


        # Lists to store patches and labels
        image_patches = []
        labels = []
        # Slice the image and mask into patches and determine labels
        for i in range(len(image_T_list)):
            tmp_image = image_T_list[i]
            tmp_mask = mask_T_list[i]
            for j in range(rows):
                for k in range(cols):
                    # Extract patch from image and mask
                    img_patch = tmp_image[:, j*patch_size[0]:(j+1)*patch_size[0], k*patch_size[1]:(k+1)*patch_size[1]]
                    mask_patch = tmp_mask[:, j*patch_size[0]:(j+1)*patch_size[0], k*patch_size[1]:(k+1)*patch_size[1]]

                    # Determine if the patch is in the area of interest
                    is_interest = mask_patch.mean().item() > 0.  # Threshold: more than 0% of pixels are interest

                    # Append patch and label
                    image_patches.append(img_patch)
                    labels.append(is_interest)

        # Convert lists to tensors
        image_patches = torch.stack(image_patches)
        labels = torch.tensor(labels)

        # Create dataset and dataloader
        dataset = TensorDataset(image_patches, labels.unsqueeze(1).float())  # Reshape labels for BCE Loss
        self.dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        self.image = image
        self.image_T_list = image_T_list
        self.patch_size = patch_size

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
