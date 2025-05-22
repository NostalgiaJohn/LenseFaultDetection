# Edge Defect Detection of Contact Lenses Based on Deep Learning

This repository contains the code and resources for the project "Edge Defect Detection of Contact Lenses Based on Deep Learning Method," submitted as a course assignment for Digital Image Processing at the School of Electronic Engineering, Sichuan University.

The project proposes a CNN-MLP network structure to identify and classify defects in the edges of contact lenses from images.

**[中文版说明 (Chinese Version)](README_CN.md)**

## Abstract

This project introduces a deep learning-based method for detecting edge defects in contact lenses. It utilizes a Convolutional Neural Network (CNN) to extract local features from image patches, followed by a Multilayer Perceptron (MLP) for defect classification. To enhance the model's generalization capabilities, several data augmentation techniques were employed, including rotation, adding Gaussian noise, and salt-and-pepper noise. Experimental results demonstrate high stability and adaptability, achieving 100% accuracy, precision, and recall on standard and rotated images. The model maintains a 97% recall under Gaussian noise conditions and a 77% recall under salt-and-pepper noise conditions, validating its potential for automated defect detection in industrial applications.

## Key Features

*   **CNN-MLP Architecture:** Combines CNN for feature extraction and MLP for classification.
*   **Patch-based Processing:** Images are divided into 25x25 patches for localized defect analysis.
*   **Data Augmentation:** Includes rotation, Gaussian noise, and salt-and-pepper noise to improve model robustness and generalization.
*   **Automated Defect Detection:** Aims to provide an efficient alternative to manual inspection.

## Methodology

1.  **Dataset Preparation:**
    *   Original contact lens images and corresponding defect masks.
    *   Images are augmented through:
        *   Rotation (90°, 180°, 270°)
        *   Addition of Gaussian noise
        *   Addition of salt-and-pepper noise
    *   A self-supplemented dataset (`image_new.bmp`, `mask_new.bmp`) is also included.
    *   Images are divided into 25x25 pixel patches. Each patch is labeled as defective or non-defective based on the mask.

2.  **Model Architecture (CNN-MLP):**
    *   **CNN Part:**
        *   Input: 1-channel 25x25 image patch.
        *   Conv1: 16 filters, kernel size 3x3, padding 1, ReLU activation.
        *   MaxPool1: kernel size 2x2, stride 2.
        *   Conv2: 32 filters, kernel size 3x3, padding 1, ReLU activation.
        *   MaxPool2: kernel size 2x2, stride 2.
        *   The output feature map is flattened (32 * 6 * 6).
    *   **MLP Part:**
        *   FC1: Linear layer (input 32\*6\*6, output 128), ReLU activation.
        *   FC2: Linear layer (input 128, output 1), Sigmoid activation for binary classification.

3.  **Training:**
    *   Optimizer: Adam (learning rate 1e-3).
    *   Loss Function: Binary Cross-Entropy Loss (BCELoss).
    *   Epochs: 100.

## Results Summary

The model was tested on various image conditions (20 rounds each):

| Test Condition         | Accuracy | Precision | Recall | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--------------------- | :------- | :-------- | :----- | :------------------------ | :------------------------ |
| Original Image         | 100%     | 100%      | 100%   | 0%                        | 0%                        |
| Rotated (90°)          | 100%     | 100%      | 100%   | 0%                        | 0%                        |
| Rotated (180°)         | 100%     | 100%      | 100%   | 0%                        | 0%                        |
| Rotated (270°)         | 100%     | 100%      | 100%   | 0%                        | 0%                        |
| Added Gaussian Noise   | 99.98%   | 97.98%    | 97%    | 0.007%                    | 3%                        |
| Added Salt&Pepper Noise| 99.91%   | 98.72%    | 77%    | 0.004%                    | 23%                       |
| Self-supplemented Image| 100%     | 100%      | 100%   | 0%                        | 0%                        |

*Note: FPR and FNR are derived from the paper's misreport and omission rates.*

## Technology Stack

*   Python 3.x
*   PyTorch
*   Torchvision
*   Pillow (PIL)
*   NumPy
*   Matplotlib

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    Ensure the training images (`image.bmp`, `mask0.bmp`, `image_new.bmp`, `mask_new.bmp`) are placed in a `train/` subfolder within the project directory. The code expects this structure:
    ```
    your-repository-name/
    ├── train/
    │   ├── image.bmp
    │   ├── mask0.bmp
    │   ├── image_new.bmp
    │   └── mask_new.bmp
    ├── dataset.py
    ├── main.py
    ├── Network.py
    ├── README.md
    └── requirements.txt
    ```

## Usage

The `main.py` script is used for training and testing the model.

1.  **Training the model:**
    *   In `main.py`, uncomment the line `instance.Train()` and `instance.save_model()`.
    *   Comment out `instance.load_model()` and `instance.Test()`.
    *   Run the script:
        ```bash
        python main.py
        ```
    *   This will train the model for 100 epochs and save the trained weights to `model.pth`.

2.  **Testing with a pre-trained model:**
    *   Ensure `model.pth` (either trained by you or provided) is in the project root.
    *   In `main.py`, ensure `instance.load_model()` and `instance.Test()` are uncommented.
    *   Comment out `instance.Train()` and `instance.save_model()`.
    *   By default, `instance.Test()` uses `self.data.image_T_list[6]` which is `image_new.bmp` for testing. You can change the index or uncomment other lines in `Test()` to test on noisy images.
    *   Run the script:
        ```bash
        python main.py
        ```
    *   The script will load the model, perform inference on the selected test image, and display the original image with detected defect regions highlighted in red.

## File Structure

*   `dataset.py`: Contains the `Data` class for loading, augmenting, and patching images.
*   `Network.py`: Defines the CNN-MLP neural network architecture.
*   `main.py`: Main script for training, testing, model saving/loading, and visualization.
*   `result.txt`: A raw summary of True Positives, True Negatives, etc., from experimental runs.
*   `model.pth`: (Generated after training) Saved model weights.
*   `train/`: Folder containing the input images and masks.
*   `requirements.txt`: Lists Python package dependencies.


## Course Information

*   **Course:** Digital Image Processing
*   **Institution:** School of Electronic Engineering, Sichuan University
