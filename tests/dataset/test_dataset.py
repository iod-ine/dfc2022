""" Unit tests for the dataset wrapper. """

import torch
import albumentations
import torchvision.transforms

from hashtagdeep.dataset import MiniFranceDFC22


def test_channel_numbers_in_the_data():
    """ The image should contain 4 channels, the label should contain 1 channel. """

    dataset = MiniFranceDFC22(
        base_dir="/home/dubrovin/Projects/Data/DFC2022",
        labeled=True,
    )

    item = dataset[13]
    image, label = item["image"], item["label"]

    assert image.shape[0] == 4, 'The number of channels in the image is different from 4'
    assert label.shape == image.shape[1:], 'Label shape does not match the image shape'


def test_returned_dtypes():
    """ Indexing the dataset should produce image tensors of type Float32 and label tensors of type Long."""

    dataset = MiniFranceDFC22(
        base_dir="/home/dubrovin/Projects/Data/DFC2022",
        labeled=True,
    )

    item = dataset[13]
    image, label = item["image"], item["label"]

    assert image.dtype == torch.float
    assert label.dtype == torch.long


def test_augmentations():
    """ Augmentations should be applied to both images and labels, or to just images if there are no labels. """

    augmentations = albumentations.Compose([
        albumentations.RandomCrop(height=256, width=256, p=1.0),
    ])

    dataset = MiniFranceDFC22(
        base_dir="/home/dubrovin/Projects/Data/DFC2022",
        labeled=True,
        augmentation=augmentations,
    )

    item = dataset[13]
    image, label = item['image'], item['label']

    assert image.shape[1:] == (256, 256), 'Incorrect image shape after RandomCrop'
    assert label.shape == (256, 256), 'Incorrect label shape after RandomCrop'

    dataset = MiniFranceDFC22(
        base_dir="/home/dubrovin/Projects/Data/DFC2022",
        labeled=False,
        augmentation=augmentations,
    )

    image = dataset[13]['image']

    assert image.shape[1:] == (256, 256), 'Incorrect image shape after RandomCrop'


def test_transforms():
    """ Transforms should be correctly applied to the images. """

    transform = torchvision.transforms.Lambda(lambda x: 'it worked :)')

    dataset = MiniFranceDFC22(
        base_dir="/home/dubrovin/Projects/Data/DFC2022",
        labeled=True,
        transform=transform,
    )

    item = dataset[13]
    image, label = item['image'], item['label']

    assert image == 'it worked :)', 'Transform was not applied to the image'
