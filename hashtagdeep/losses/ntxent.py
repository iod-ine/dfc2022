""" NTXent contrastive loss for SimCLR. """

import torch
import torch.nn as nn
import torch.nn.functional as func


class NTXentLoss(nn.Module):
    """ Normalized Temperature-scaled Cross-Entropy (NTXent) loss for SimCLR. """

    def __init__(self, temperature: float):
        """ Initialize a new NTXent module.

        Args:
            temperature (float): Temperature scaling parameter.

        """
        super().__init__()
        self.temperature = temperature

    def forward(self, stacked_batch):
        """ Calculate the loss for a stacked batch.
        
        Args:
            stacked_batch (torch.Tensor): Batch of augmented pairs with shape (batch_size * 2, hidden_dim).

        Notes:
            The positive pairs in the stacked_batch are assumed to be batch_size elements apart:
            stacked_batch = torch.cat([features_from_augmented0, features_from_augmented1], dim=0)

        """

        similarity = func.cosine_similarity(stacked_batch[:, None, :], stacked_batch[None, :, :], dim=-1)
        similarity = similarity / self.temperature

        self_mask = torch.eye(similarity.shape[0], dtype=bool, device=similarity.device)
        positive_mask = self_mask.roll(shifts=similarity.shape[0] // 2, dims=0)

        similarity.masked_fill_(self_mask, -9e15)  # self-similarities need not apply

        loss = torch.logsumexp(similarity, dim=-1) - similarity[positive_mask]

        return loss.mean()
