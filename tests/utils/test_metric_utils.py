""" Unit test for metric helper functions. """

import torch

from hashtagdeep.utils.metrics import intersection_union_target


def test_intersection_union_target():
    """ Compare the results of intersection_union_target with manual calculations on simple images. """

    pred = torch.zeros(4, 4, 4)  # num_classes x height x width

    pred[0, :2, :2] = 1
    pred[1, :2, 2:] = 1
    pred[2, 2:, :2] = 1
    pred[3, 2:, 2:] = 1

    pred.unsqueeze_(dim=0)  # batch_size x num_classes x height x width

    mask0 = torch.zeros(1, 4, 4) + 3
    mask1 = torch.zeros(1, 4, 4) + 3
    mask2 = torch.arange(2).repeat([4, 2]).unsqueeze(dim=0)

    mask0[0, :, :2] = 2
    mask0[0, :2, :] = 1
    mask0[0, :2, :2] = 0

    mask1[0, :, :2] = 3
    mask1[0, :2, :] = 2
    mask1[0, :2, :2] = 1

    i0, u0, t0 = intersection_union_target(pred, mask0)
    i1, u1, t1 = intersection_union_target(pred, mask1)
    i2, u2, t2 = intersection_union_target(pred, mask2)

    assert torch.all(i0 == torch.tensor([[4, 4, 4, 4]])), i0
    assert torch.all(u0 == torch.tensor([[4, 4, 4, 4]])), u0
    assert torch.all(t0 == torch.tensor([[4, 4, 4, 4]])), t0

    assert torch.all(i1 == torch.tensor([[0, 0, 0, 4]])), i1
    assert torch.all(u1 == torch.tensor([[4, 8, 8, 8]])), u1
    assert torch.all(t1 == torch.tensor([[0, 4, 4, 8]])), t1

    assert torch.all(i2 == torch.tensor([[2, 2, 0, 0]])), i2
    assert torch.all(u2 == torch.tensor([[10, 10, 4, 4]])), u2
    assert torch.all(t2 == torch.tensor([[8, 8, 0, 0]])), t2
