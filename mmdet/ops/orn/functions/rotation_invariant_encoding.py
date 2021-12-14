# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from .. import orn_cuda

class _RotationInvariantEncoding(Function):
  @staticmethod
  def forward(ctx, input, nOrientation, return_direction=False):
    ctx.nOrientation = nOrientation
    ctx.return_direction = return_direction
    mainDirection, output = orn_cuda.rie_forward(input, nOrientation)
    if return_direction:
      ctx.save_for_backward(input, mainDirection)
      ctx.mark_non_differentiable(mainDirection)
      return output, mainDirection
    else:
      ctx.save_for_backward(input)
      ctx.mainDirection = mainDirection
      return output

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_output):
    if ctx.return_direction:
      input, mainDirection = ctx.saved_tensors
    else:
      input, = ctx.saved_tensors
      mainDirection = ctx.mainDirection
    grad_input = orn_cuda.rie_backward(mainDirection, grad_output, ctx.nOrientation)
    return grad_input, None, None


rotation_invariant_encoding = _RotationInvariantEncoding.apply


class RotationInvariantEncoding(nn.Module):
  def __init__(self, nOrientation, return_direction=False):
    super(RotationInvariantEncoding, self).__init__()
    self.nOrientation = nOrientation
    self.return_direction = return_direction

  def forward(self, input):
    return rotation_invariant_encoding(input, self.nOrientation, self.return_direction)


if __name__ == '__main__':
  nOrientation = 8
  input = torch.randn(2,8,1,1).double()#.cuda()
  input.requires_grad = True
  output = rotation_invariant_encoding(input, nOrientation)
  # check grad
  res = torch.autograd.gradcheck(rotation_invariant_encoding, (input, nOrientation), raise_exception=True)
  print(res)
