""" A PyTorch wrapper for the 2022 IEEE GRSS Data Fusion Contest data. """

import os
import pathlib

import rasterio
import numpy as np
import torch.utils.data
import torch.nn.functional as func


class MiniFranceDFC22(torch.utils.data.Dataset):

    def __init__(self, base_dir, labeled=True, val=False, augmentation=None, transform=None):
        """ Create a new MiniFranceDFC22 instance.
        
        Args:
            base_dir: Path to the root of the dataset (that contains labeled_train, unlabeled_train, and val folders).
            labeled (bool): Whether to load the labeled or unlabeled part of the dataset.
            val (bool): Whether to load validation or train part (for unlabeled only).
            augmentation: Augmentation to apply. Assumed to be an albumentations.Compose instance.
            transform: Transform to apply to the image. Assumed to be a torchvision.transforms.Compose instance.

        """

        self.base_dir = pathlib.Path(base_dir)
        self.labeled = labeled
        self.val = val
        self.augmentation = augmentation
        self.transform = transform

        self.true_color_paths = []
        self.elevation_paths = []
        self.label_paths = []

        if labeled:
            subdir = self.base_dir / 'labeled_train'
        else:
            subdir = self.base_dir / 'val' if val else self.base_dir / 'unlabeled_train'

        regions = [r for r in os.listdir(subdir) if not r.startswith('.')]

        for region in regions:
            tci_base_dir = subdir / region / 'BDORTHO'
            images = [tci_base_dir / f for f in os.listdir(tci_base_dir) if '.tif' in f]
            self.true_color_paths.extend(images)

        self.true_color_paths.sort()

        for tci_path in self.true_color_paths:
            dem_path = str(tci_path).replace('BDORTHO', 'RGEALTI')
            dem_path = dem_path.replace('.tif', '_RGEALTI.tif')
            self.elevation_paths.append(pathlib.Path(dem_path))

            if labeled:
                mask_path = str(tci_path).replace('BDORTHO', 'UrbanAtlas')
                mask_path = mask_path.replace('.tif', '_UA2012.tif')
                self.label_paths.append(pathlib.Path(mask_path))

    def __len__(self):
        """ Get the number of samples in the dataset. """

        return len(self.true_color_paths)

    def __getitem__(self, idx):
        """ Load and return a sample from the dataset at the given index. """

        if self.labeled:
            label_path = self.label_paths[idx]

            with rasterio.open(label_path) as data:
                label = data.read().astype(np.int64)

            label = label.squeeze(axis=0)
            label[label == 15] = 0  # there are some sneaky 15s in at least one image
            label = label - 1  # the label should represent the class index. index -1 will be set as ignored

        tci_path = self.true_color_paths[idx]
        dem_path = self.elevation_paths[idx]

        with rasterio.open(tci_path) as data:
            tci = data.read().astype(np.float32)

        with rasterio.open(dem_path) as data:
            dem = data.read()

        # elevation data has spatial resolution of 100 cm vs. 50 cm for TCI and masks
        dem[dem == -99999.0] = 0  # upsampling will smear the no-data values into a mess of negative elevations
        dem = torch.from_numpy(dem)
        dem.unsqueeze_(dim=0)
        dem = func.interpolate(dem, size=tci.shape[-2:], mode='bilinear', align_corners=False)
        dem.squeeze_(dim=0)
        dem = dem.numpy()

        image = np.concatenate([tci, dem], axis=0)

        if self.augmentation is not None:
            image = np.transpose(image, (1, 2, 0))  # albumentations wants images in HWC format
            if self.labeled:
                augmented = self.augmentation(image=image, mask=label)
                image, label = augmented['image'], augmented['mask']
            else:
                augmented = self.augmentation(image=image)
                image = augmented['image']
            image = np.transpose(image, (2, 0, 1))  # PyTorch expects CHW format

        image = torch.from_numpy(image)

        if self.transform is not None:
            image = self.transform(image)

        item = {'image': image}

        if self.labeled:
            item['label'] = torch.from_numpy(label)

        return item
