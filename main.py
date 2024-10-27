from dataset import Data
from Network import Network

import torch.nn as nn
import torch.optim as optim


class Process:
    def __init__(self):
        # Load data
        data = Data()
        self.dataloader = data.dataloader

        # Instantiate model, define loss and optimizer
        self.model = Network()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def Train(self):
        # Training loop
        num_epochs = 100

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.dataloader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update weights

                running_loss += loss.item()

            # Print epoch loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.dataloader):.4f}")

if __name__ == "__main__":
    process = Process()
    process.Train()
