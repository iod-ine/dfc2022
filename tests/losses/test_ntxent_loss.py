""" Unit tests for NTXent contrastive loss used in SimCLR. """

import torch

from hashtagdeep.losses import NTXentLoss


def manual_ntxent_loss(stacked_batch, temperature: float = 2.0):
    """ Calculate NTXent loss manually.

    Args:
        stacked_batch (torch.Tensor): Batch of augmented pairs with shape (batch_size * 2, hidden_dim).
        temperature (float): Temperature scaling parameter.

    Notes:
        The positive pairs in the stacked_batch are assumed to be batch_size elements apart:
        stacked_batch = torch.cat([features_from_augmented0, features_from_augmented1], dim=0)

    """

    batch = stacked_batch.shape[0] // 2
    loss_ij = torch.empty(stacked_batch.shape[0])  # calculated for all positive pairs: (i, j) and (j, i)

    for i in range(stacked_batch.shape[0]):
        j = i + batch if i < batch else i - batch  # the positive pair for i
        similarity = torch.cosine_similarity(stacked_batch[i], stacked_batch[j], dim=0) / temperature
        sum_exp = 0
        for k in range(stacked_batch.shape[0]):
            if k == i:
                continue
            _sim = torch.cosine_similarity(stacked_batch[i], stacked_batch[k], dim=0) / temperature
            sum_exp += torch.exp(_sim)
        log_sum_exp = torch.log(sum_exp)
        loss_ij[i] = log_sum_exp - similarity

    return loss_ij.mean()


def test_ntxent_loss():
    """ Vectorized Module loss should be the same as the manually calculated one. """

    ntxent = NTXentLoss(temperature=2.0)

    torch.manual_seed(42)

    for i in range(10):
        random_stacked_batch = torch.randn(8, 16)  # batch_size = 4, hidden_dim = 16
        manual = manual_ntxent_loss(random_stacked_batch)
        module = ntxent(random_stacked_batch)
        assert torch.isclose(manual, module)
