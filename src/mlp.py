import torch
import torch.nn as nn
import torch.nn.functional as F

class OurMLP(nn.Module):
  def __init__(self, num_classes=1, in_channels=1):
    super(OurMLP,self).__init__()

    self.mlp = nn.Sequential(
      nn.Linear(in_channels, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, num_classes),
    )
            
  def forward(self,x):
    x = self.mlp(x)
    
    return x