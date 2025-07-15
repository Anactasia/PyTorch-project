import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # вычислим размер после сверток и пулинга для 128x128 входа:
        # 128 -> pool -> 64 -> pool -> 32 -> pool -> 16
        self._flatten_dim = self._get_flatten_size()

        self.fc1 = nn.Linear(self._flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128) 
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
