import torch
import torch.nn as nn

class PackageClassifier(nn.Module):
    def __init__(self):
        super(PackageClassifier, self).__init__()
        self.input_size = 4  # Number of input features (box width, box length, area, ratio)
        self.hidden_size = 64
        self.num_classes = 6
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out