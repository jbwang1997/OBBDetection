import torch
from ..builder import ANCHOR_GENERATORS
from ..anchor_generator import AnchorGenerator


@ANCHOR_GENERATORS.register_module()
class Theta0AnchorGenerator(AnchorGenerator):

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        anchors = super(Theta0AnchorGenerator, self).single_level_grid_anchors(
            base_anchors, featmap_size, stride=stride, device=device)

        num_anchors = anchors.size(0)
        xy = (anchors[:, 2:] + anchors[:, :2]) / 2
        wh = anchors[:, 2:] - anchors[:, :2]
        theta = xy.new_zeros((num_anchors, 1))

        anchors = torch.cat([xy, wh, theta], axis=1)
        return anchors
