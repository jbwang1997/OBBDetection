# Poly IoU Loss

## Rotated IoU Loss

IoULoss is a very common-used loss function in object detection. It can directly optimize IoU between predictions and targets. However, the IoU calculation between oriented boxes is very complicated so that it usually undifferentiable.

Recently, [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU) implements a differentiable IoU calculation. It first gets all vertices of interstions, then sorts them and calculates the areas by shoelace formula. Following its way, [s2anet](https://github.com/csuhan/s2anet/blob/master/configs/rotated_iou/README.md) applies `RotatedIoULoss` in oriented detectors and further boost the performance.

## Our implement

Following the strategy of `RotatedIoULoss`, I implement the `PolyIoULoss` in OBBDetection. Its steps are quite similar to `RotatedIoULoss`, but I discard the prior condition that two joint sides of boxes are perpendicular. There are six steps to calculate Poly IoU.

+ Use vertices to represent boxes.
+ Calculate the intersection points of two sides of different Polys.
+ Judge whether the vertices of one Poly locate in the other one. If true, this point also is a vertice of the intersection.
+ Sort all points in counterclockwise order.
+ Calculate the intersection areas of Polys using the shoelace formula.
+ Finally, IoU = intersections / (areas1 - areas2 + intersections)

The code of `PolyIoULoss` is [here](https://github.com/jbwang1997/OBBDetection/blob/master/mmdet/models/losses/obb/poly_iou_loss.py).


**note**: The `PolyIoULoss` cannot calculate arbitrary polygons IoU. It only calculates the IoU between two convex polygons.

## Results

I test the `PolyIoULoss` in [RetinaNet OBB](https://github.com/jbwang1997/OBBDetection/tree/master/configs/obb/retinanet_obb).

|     model     | backbone | dataset | origin loss | origin mAP | new mAP |
|:-------------:|:--------:|:-------:|:-----------:|:----------:|:-------:|
| RetinaNet OBB |  R50-FPN | DOTA1.0 |      L1     |    69.4%   |  70.6%  |
