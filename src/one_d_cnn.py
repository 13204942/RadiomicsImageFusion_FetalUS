import torch
import torch.nn as nn
import torch.nn.functional as F

class OneDCNN(nn.Module):
  def __init__(self, num_classes=1, in_channels=1, out_channels=64, kernel_size=3, stride=1, dropout=0.25):
    super(OneDCNN,self).__init__()
    self.ConvNet = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size),
      nn.ReLU(),
      nn.BatchNorm1d(16),
      nn.MaxPool1d(kernel_size=2),

      nn.Conv1d(in_channels=16, out_channels=64, kernel_size=kernel_size),
      nn.ReLU(),
      nn.BatchNorm1d(64),
      nn.MaxPool1d(kernel_size=2),

      nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size),
      nn.ReLU(),
      nn.BatchNorm1d(out_channels),
      nn.MaxPool1d(kernel_size=2),

      nn.Dropout(dropout),
      nn.Flatten(),
    )
    self.fc = nn.Sequential(
      nn.Linear(640, 100),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, num_classes),
    )
        
  def forward(self,x):
    x = self.ConvNet(x)
    x = self.fc(x)
        
    return x