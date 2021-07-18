import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
  
  def __init__(self, h, w, outputs):
    super(Network, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, stride = 2)
    self.bn3 = nn.BatchNorm2d(32)

    convw = Network.conv2d_size_out(Network.conv2d_size_out(Network.conv2d_size_out(w)))
    convh = Network.conv2d_size_out(Network.conv2d_size_out(Network.conv2d_size_out(h)))

    self.head = nn.Linear(convw * convh * 32, outputs)
    
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
  
    return self.head(x.view(x.size(0), -1))

  @staticmethod
  def conv2d_size_out(size, kernel_size = 5, stride = 2):
    return (size - (kernel_size - 1) - 1) // stride + 1
