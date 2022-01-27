""" This is a template for an experiment. """

import torch
import albumentations
import torch.nn as nn
import torch.utils.data
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import pytorch_lightning.loggers as loggers
import sklearn.model_selection as model_selection

from hashtagdeep.utils import metrics
from hashtagdeep.models import FCDenseNet103
from hashtagdeep.dataset import MiniFranceDFC22
from hashtagdeep.utils import colormaps as colors
from hashtagdeep.utils import visualization_utils as visualize


class ExperimentSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.net = FCDenseNet103(in_channels=4, n_classes=14)

        # self.loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)  # v0
        # self.loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=-1)  # v1
        self.loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=-1)  # v2

        # a tiny offset to use in area-based metric calculation to avoid dividing by zero
        self._epsilon = 1e-8

    def forward(self, x):
        logits = self.net(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        pred = self.forward(x)
        loss = self.loss(pred, y)
        return {'loss': loss}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = [x['loss'] for x in train_step_outputs]
        avg_train_loss = torch.tensor(avg_train_loss).mean()
        self.log('avg_loss/train', avg_train_loss)

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']

        # split the prediction for a large validation image into parts
        b, _, h, w = x.shape
        pred = torch.empty(b, 14, h, w, device=self.device)

        window = 256
        width_steps = w // window + 1
        height_steps = h // window + 1

        for i in range(height_steps):
            for j in range(width_steps):
                w0, w1 = window * j, window * (j + 1)
                h0, h1 = window * i, window * (i + 1)

                if w1 > w:
                    w0, w1 = w - window, w

                if h1 > h:
                    h0, h1 = h - window, h

                pred[:, :, h0:h1, w0:w1] = self(x[:, :, h0:h1, w0:w1])

        loss = self.loss(pred, y)
        intersection, union, target = metrics.intersection_union_target(pred, y)

        intersection_over_union = intersection / (union + self._epsilon)
        recall = intersection / (target + self._epsilon)
        accuracy = intersection.sum(dim=1) / target.sum(dim=1)

        out = {
            'loss': loss,
            'mIoU': intersection_over_union.mean(),
            'mRecall': recall.mean(),
            'accuracy': accuracy.mean(),
            'prediction': pred.argmax(dim=1),
        }

        return out

    def validation_epoch_end(self, val_step_outputs):
        """ At the end of every validation epoch, calculate and log all metrics and images. """

        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()

        intersection_over_union = torch.tensor([x['mIoU'] for x in val_step_outputs])
        recall = torch.tensor([x['mRecall'] for x in val_step_outputs])
        accuracy = torch.tensor([x['accuracy'] for x in val_step_outputs])

        self.log('mIoU', intersection_over_union.mean(), sync_dist=True)
        self.log('mRecall', recall.mean(), sync_dist=True)
        self.log('Accuracy', accuracy.nanmean(), sync_dist=True)
        self.log('avg_loss/val', avg_val_loss, sync_dist=True)

        preds = torch.cat([x['prediction'][:, 744:1256, 744:1256] for x in val_step_outputs])
        pred = visualize.make_image_tensor_for_segmentation_mask(preds, colors.dfc22_labels_color_map)

        self.logger.experiment.add_images('Prediction', pred, self.current_epoch)

    def on_train_start(self):
        """ Before training starts, log the ground truth images for the validation data. """

        val_dataloader = self.trainer.datamodule.val_dataloader()
        tci = torch.cat([x['image'][:, :3, 744:1256, 744:1256] for x in val_dataloader]) / 255
        masks = torch.cat([x['label'][:, 744:1256, 744:1256] for x in val_dataloader])

        masks = visualize.make_image_tensor_for_segmentation_mask(masks, colors.dfc22_labels_color_map)

        self.logger.experiment.add_images('Images', tci, 0)
        self.logger.experiment.add_images('Ground Truth', masks, 0)

        self.logger.experiment.add_text('Indices of validation samples', str(self.trainer.datamodule.val_indices))


class ExperimentDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=4):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size

        self.augmentations = albumentations.Compose([
            albumentations.RandomCrop(height=256, width=256, p=1.0),
            albumentations.RandomRotate90(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
        ])

        self.transforms = None

        self._train_dataset = MiniFranceDFC22(
            base_dir=self.data_dir,
            labeled=True,
            augmentation=self.augmentations,
            transform=self.transforms,
        )

        self._val_dataset = MiniFranceDFC22(
            base_dir=self.data_dir,
            labeled=True,
            augmentation=None,
            transform=self.transforms,
        )

        self.train_indices = None
        self.val_indices = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage=None):
        all_indices = range(len(self._train_dataset))
        self.train_indices, self.val_indices = model_selection.train_test_split(all_indices, test_size=0.07)
        self.train_data = torch.utils.data.Subset(self._train_dataset, self.train_indices)
        self.val_data = torch.utils.data.Subset(self._val_dataset, self.val_indices)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=True,
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=6,
            persistent_workers=True,
        )
        return loader


if __name__ == '__main__':
    best_val_miou = pl.callbacks.ModelCheckpoint(
        monitor='mIoU',
        mode='max',
        every_n_epochs=1,
        save_top_k=3,
        dirpath='./checkpoints/experiment00/v2',
        filename='experiment00_miou={mIoU:.4f}',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_last=True,
    )

    best_val_recall = pl.callbacks.ModelCheckpoint(
        monitor='mRecall',
        mode='max',
        every_n_epochs=1,
        save_top_k=3,
        dirpath='./checkpoints/experiment00/v2',
        filename='experiment00_recall={mRecall:.4f}',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_last=True,
    )

    best_val_acc = pl.callbacks.ModelCheckpoint(
        monitor='Accuracy',
        mode='max',
        every_n_epochs=1,
        save_top_k=3,
        dirpath='./checkpoints/experiment00/v2',
        filename='experiment00_acc={Accuracy:.4f}',
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    bast_avg_loss = pl.callbacks.ModelCheckpoint(
        monitor='avg_loss/val',
        mode='min',
        every_n_epochs=1,
        save_top_k=3,
        dirpath='./checkpoints/experiment00/v2',
        filename='experiment00_loss={avg_loss/val:.4f}',
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    logger = loggers.TensorBoardLogger(
        save_dir='lightning_logs',
        name='experiment00',
    )

    model = ExperimentSystem()
    dm = ExperimentDataModule(data_dir='/home/dubrovin/Projects/Data/DFC2022')

    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        fast_dev_run=False,
        max_epochs=300,
        callbacks=[best_val_miou, best_val_recall, best_val_acc, bast_avg_loss],
        # auto_scale_batch_size=True,
    )

    trainer.fit(model, datamodule=dm)
