""" A collection of utility functions to use for visualization. """


def make_image_tensor_for_segmentation_mask(masks, colormap):
    """ Transform the predictions for semantic segmentation into an image tensor to pass to a SummaryWriter.

    Args:
        masks (torch.tensor): Predictions or ground truth masks with shape (batch_size, height, width).
        colormap (torch.tensor): A map between class indices and rgb colors with shape (num_classes, 3).

    Returns:
        image_tensor: RGB tensor with shape

    Notes:
        To make this function universal for both the predicted and ground truth masks, it's assumed that the predictions
        are converted to class indices with argmax(dim=1).

    """

    batch, height, width = masks.shape
    flat_masks = masks.view(-1)

    if colormap.device != masks.device:
        colormap.to(masks.device)

    rgb = colormap[flat_masks].reshape(batch, height, width, 3).permute(0, 3, 1, 2) / 255

    return rgb
