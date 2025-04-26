# Import the PyTorch library
import torch
import torch.nn as nn

# Define a Neural Network class that inherits from nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        # First fully connected layer: input to hidden
        self.l1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer: hidden to hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Third fully connected layer: hidden to output
        self.l3 = nn.Linear(hidden_size, output_size)
        # Activation function (ReLU)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through first layer, then apply ReLU activation
        out = self.l1(x)
        out = self.relu(out)
        # Pass through second layer and apply ReLU activation
        out = self.l2(out)
        out = self.relu(out)
        # Pass through final output layer
        out = self.l3(out)
        return out  # Return the output
