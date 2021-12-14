# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from .. import orn_cuda
#import _C


class _ActiveRotatingFilter(Function):
  @staticmethod
  def forward(ctx, input, indices):
    indices = indices.byte()
    ctx.input = input
    output = orn_cuda.arf_forward(input, indices)
    ctx.save_for_backward(indices)
    return output

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_output):
    indices, = ctx.saved_tensors
    input = ctx.input
    grad_input = orn_cuda.arf_backward(indices, grad_output)
    return grad_input, None


active_rotating_filter = _ActiveRotatingFilter.apply


class ActiveRotatingFilter(nn.Module):
  def __init__(self, indices):
    super(ActiveRotatingFilter, self).__init__()
    self.indices = indices

  def forward(self, input):
    return active_rotating_filter(input, self.indices)


if __name__ == "__main__":

  import math
  def get_indices(nOrientation, nRotation, kernel_size, mode='fast'):
    kernel_indices = {
      1: {
        0: (1,),
        45: (1,),
        90: (1,),
        135: (1,),
        180: (1,),
        225: (1,),
        270: (1,),
        315: (1,)
      },
      3: {
        0: (1,2,3,4,5,6,7,8,9),
        45: (2,3,6,1,5,9,4,7,8),
        90: (3,6,9,2,5,8,1,4,7),
        135: (6,9,8,3,5,7,2,1,4),
        180: (9,8,7,6,5,4,3,2,1),
        225: (8,7,4,9,5,1,6,3,2),
        270: (7,4,1,8,5,2,9,6,3),
        315: (4,1,2,7,5,3,8,9,6)
      }
    }
    delta_orientation = 360 / nOrientation
    delta_rotation = 360 / nRotation
    kH, kW = kernel_size
    indices = torch.ByteTensor(nOrientation * kH * kW, nRotation)
    for i in range(0, nOrientation):
      for j in range(0, kH * kW):
        for k in range(0, nRotation):
          angle = delta_rotation * k
          layer = (i + math.floor(angle / delta_orientation)) % nOrientation
          kernel = kernel_indices[kW][angle][j]
          indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
    return indices.view(nOrientation, kH, kW, nRotation)

  out_channels = 4
  in_channels = 2
  nOrientation = 8
  nRotation = 8
  kernel_size = 3
  input = torch.randn(out_channels, in_channels, nOrientation, kernel_size, kernel_size)
  input.requires_grad = True
  input = input.double()
  indices = get_indices(nOrientation, nRotation, (kernel_size, kernel_size))
  input = input.cuda()
  indices = indices.cuda()
  output = active_rotating_filter(input, indices)
  print(output.size())
  res = torch.autograd.gradcheck(active_rotating_filter, (input, indices), raise_exception=True)
  print(res)
