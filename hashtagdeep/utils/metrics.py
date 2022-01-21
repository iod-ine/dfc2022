""" Helper functions for metric calculation. """

import torch


def intersection_union_target(predictions, masks):
    """ Calculate the intersection and union for mIoU and target for recall and accuracy.

    Args:
        predictions: Logits or probabilities of shape (batch_size x num_classes x height x width).
        masks: Ground truth masks for the predictions of shape (batch_size x height x width).

    Notes:
        The values of the masks are class indices (in range [0, num_classes - 1]).
        When calculating mIoU, add a small offset (i.e., eps=1e-7) to avoid problems with inevitable zeros.

    Returns:
        intersection, union, target for each class. Target is the number of pixels of that class in the image.
        All have shape (batch_size x num_classes).

    """

    batch_size, num_classes, _, _ = predictions.shape
    predictions = predictions.argmax(dim=1)

    intersection = torch.empty(batch_size, num_classes, dtype=torch.int32)
    union = torch.empty(batch_size, num_classes, dtype=torch.int32)
    target = torch.empty(batch_size, num_classes, dtype=torch.int32)

    for i in range(num_classes):
        intersection[:, i] = torch.logical_and(predictions == i, masks == i).count_nonzero(dim=(1, 2))
        union[:, i] = torch.logical_or(predictions == i, masks == i).count_nonzero(dim=(1, 2))
        target[:, i] = torch.count_nonzero(masks == i, dim=(1, 2))

    return intersection, union, target
