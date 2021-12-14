import torch
from torch import nn
from torch.nn import functional as F


class RotationInvariantPooling(nn.Module):
  def __init__(self, nInputPlane, nOrientation=8):
    super(RotationInvariantPooling, self).__init__()
    self.nInputPlane = nInputPlane
    self.nOrientation = nOrientation
    
    # hiddent_dim = int(nInputPlane / nOrientation)
    # self.conv = nn.Sequential(
      # nn.Conv2d(hiddent_dim, nInputPlane, 1, 1),
      # nn.BatchNorm2d(nInputPlane),
    # )

  def forward(self, x):
    # x: [N, c, 1, w]
    ## first, max_pooling along orientation.
    N, c, h, w = x.size()
    x = x.view(N, -1, self.nOrientation, h, w)
    x, _ = x.max(dim=2, keepdim=False) # [N, nInputPlane/nOrientation, 1, w]
    # MODIFIED
    # x = self.conv(x) # [N, nInputPlane, 1, w]
    return x


if __name__ == '__main__':
  inst = RotationInvariantPooling(512, 8)
  input = torch.randn(8, 512, 1, 25)
  output = inst(input)
  print(output.size())
