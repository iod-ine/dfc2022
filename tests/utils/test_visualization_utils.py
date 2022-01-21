""" Unit tests for visualization utility functions. """

import torch

from hashtagdeep.utils import visualization_utils as viz
from hashtagdeep.utils.colormaps import dfc22_labels_color_map


def test_make_image_tensor_for_segmentation_mask():
    """ This  should transform a 1 channel tensor with predicted class indices to a 3 channel RGB image tensor. """

    prediction = torch.tensor([
        [9, 11, 12],  # Class 10 (Forests), Class 12 (Open space), Class 13 (Wetlands)
        [0, 1, 2],  # Class 1 (Urban), Class 2 (Industrial), Class 3 (Construction)
        [13, 4, 6],  # Class 14 (Water), Class 5 (Annual crops), Class 7 (Pastures)
    ])

    # this image was tested manually with: plt.imshow(image.permute(1, 2, 0) / 255);
    image = torch.tensor([
        [  # red channel
            [55., 145., 102.],
            [204., 209., 218.],
            [34., 143., 111.],
        ],
        [  # green channel
            [125., 95., 155.],
            [102., 153., 207.],
            [102., 215., 174.],
        ],
        [  # blue channel
            [34., 38., 214.],
            [92., 98., 106.],
            [246., 105., 98.],
        ],
    ])

    image /= 255

    transformed = viz.make_image_tensor_for_segmentation_mask(
        masks=prediction.unsqueeze(0),
        colormap=dfc22_labels_color_map,
    )

    assert torch.all(image == transformed.squeeze(0))
