import torch
import torch.nn as nn
import torch.nn.functional as F

class OneAttention(nn.Module):
  def __init__(self, num_classes=1, in_channels=1, stride=1, dropout=0.25):
    super().__init__()
    self.ConvNet = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=1),
      nn.ReLU(),
      nn.BatchNorm1d(32),
      nn.MaxPool1d(kernel_size=1)
    )
    
    self.attention_layer = AttentionLayer(32)
    self.flatten = nn.Flatten()
    self.fc = nn.Sequential(
      nn.Linear(3040, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, num_classes)
    )
        
  def forward(self,x):
    x = self.ConvNet(x)
    # x = self.fc(x)
    x = x.transpose(1,2)
    x = self.attention_layer(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x


class AttentionLayer(nn.Module):
  def __init__(self, embed_dim):
    super(AttentionLayer, self).__init__()
    self.query_layer = nn.Linear(embed_dim, embed_dim)
    self.key_layer = nn.Linear(embed_dim, embed_dim)
    self.value_layer = nn.Linear(embed_dim, embed_dim)
    self.scale_factor = embed_dim ** 0.5  # Square root of embed dimension for scaling
  
  def forward(self, x):
    Q = self.query_layer(x)
    K = self.key_layer(x)
    V = self.value_layer(x)
    
    # Compute the scaled dot-product attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
    attention_weights = F.softmax(scores, dim=-1)
    
    # Output weighted sum of values
    attention_output = torch.matmul(attention_weights, V)
    
    return attention_output