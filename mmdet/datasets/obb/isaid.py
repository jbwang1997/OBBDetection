from ..builder import DATASETS
from ..coco import CocoDataset


@DATASETS.register_module()
class ISAIDDataset(CocoDataset):

    CLASSES = ('ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
               'basketball_court', 'Ground_Track_Field', 'Bridge',
               'Large_Vehicle', 'Small_Vehicle', 'Helicopter', 'Swimming_pool',
               'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor')

    def pre_pipeline(self, results):
        results['cls'] = self.CLASSES
        super().pre_pipeline(results)
