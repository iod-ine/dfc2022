""" Unit tests for FC-DenseNet103Encoder implementation. """

import pytest
import torch
import torch.nn.functional
import torch.backends.cudnn

from hashtagdeep.models import FCDenseNet103Encoder
from hashtagdeep.losses import NTXentLoss


@pytest.fixture(scope='function')
def model():
    model = FCDenseNet103Encoder(in_channels=4)
    return model


@pytest.fixture(scope='module')
def ntxent():
    return NTXentLoss(temperature=0.5)


def test_parameters_change_after_learning_step(model, ntxent):
    """ If the parameters are not explicitly frozen, they should change after one learning step. """

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    params_before = [(name, data.detach().clone()) for name, data in model.named_parameters()]

    xx = torch.ones(4, 4, 256, 256)
    yy = model(xx)
    loss = ntxent(yy)
    loss.backward()
    optimizer.step()

    params_after = model.parameters()

    for before, after in zip(params_before, params_after):
        assert torch.any(before[1] != after), f'Parameter {before[0]} did not change'


def test_loss_decreases_after_learning_step(model, ntxent):
    """ The loss should decrease after each learning step. """

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    xx = torch.ones(1, 4, 256, 256)
    yy = model(xx)
    loss0 = ntxent(yy)
    loss0.backward()
    optimizer.step()

    yy = model(xx)
    loss1 = ntxent(yy)

    assert loss1.item() <= loss0.item(), 'Loss increased after a learning step'


@torch.no_grad()
def test_loss_is_not_zero(model, ntxent):
    """ The loss should never be zero. """

    xx = torch.ones(1, 4, 256, 256)
    yy = model(xx)
    loss = torch.nn.functional.mse_loss(yy, torch.ones(1, 128))

    assert loss.item() != 0, f'Loss is zero'


@torch.no_grad()
def test_logits_output_range(model):
    """ Logits should have values both above and below zero. """

    xx = torch.ones(1, 4, 256, 256)
    yy = model(xx)

    assert torch.any(yy.max() > 0), 'No positive values in the logit output'
    assert torch.any(yy.min() < 0), 'No negative values in the logit output'


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available.')
def test_switch_devices(model):
    """ There should be no problems with device switching. """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()

    xx = torch.ones(2, 4, 256, 256)

    torch.manual_seed(13)
    yy = model(xx)

    torch.manual_seed(13)
    model.to('cuda:0')
    yy_gpu = model(xx.to('cuda:0'))

    torch.manual_seed(13)
    model.to('cpu')
    yy_cpu = model(xx)

    assert torch.allclose(yy, yy_gpu.to('cpu'), atol=1e-6), 'Output on GPU significantly differs from output on CPU'
    assert torch.allclose(yy, yy_cpu), 'Output on CPU changed after model switched devices twice'
