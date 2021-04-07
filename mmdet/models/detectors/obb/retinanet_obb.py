from mmdet.models.builder import DETECTORS
from .obb_single_stage import OBBSingleStageDetector


@DETECTORS.register_module()
class RetinaNetOBB(OBBSingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetOBB, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained)
