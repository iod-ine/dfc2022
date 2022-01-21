""" Unit tests for FC-DenseNet103 implementation. """

import pytest
import torch
import torch.nn.functional
import torch.backends.cudnn

from hashtagdeep.models import FCDenseNet103


@pytest.fixture(scope='function')
def model():
    model = FCDenseNet103(in_channels=4, n_classes=10)
    return model


def test_learnable_parameters_change_after_learning_step(model):
    """ If the parameters are not explicitly frozen, they should change after one learning step. """

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    params_before = [(name, data.detach().clone()) for name, data in model.named_parameters()]

    xx = torch.ones(1, 4, 255, 257)
    yy = model(xx)
    loss = torch.nn.functional.cross_entropy(yy, torch.ones(1, 255, 257, dtype=torch.long))
    loss.backward()
    optimizer.step()

    params_after = model.parameters()

    for before, after in zip(params_before, params_after):
        assert torch.any(before[1] != after), f'Parameter {before[0]} did not change'


def test_loss_decreases_after_learning_step(model):
    """ The loss should decrease after each learning step. """

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    xx = torch.ones(1, 4, 255, 257)
    yy = model(xx)
    loss0 = torch.nn.functional.cross_entropy(yy, torch.ones(1, 255, 257, dtype=torch.long))
    loss0.backward()
    optimizer.step()

    yy = model(xx)
    loss1 = torch.nn.functional.cross_entropy(yy, torch.ones(1, 255, 257, dtype=torch.long))

    assert loss1.item() <= loss0.item(), 'Loss increased after a learning step'


@torch.no_grad()
def test_loss_is_not_zero(model):
    """ The loss should never be zero. """

    xx = torch.ones(1, 4, 255, 257)
    yy = model(xx)
    loss = torch.nn.functional.cross_entropy(yy, torch.ones(1, 255, 257, dtype=torch.long))

    assert loss.item() != 0, f'Loss is zero'


@torch.no_grad()
def test_logits_output_range(model):
    """ Logits should have values both above and below zero. """

    xx = torch.ones(1, 4, 255, 257)
    yy = model(xx)

    assert torch.any(yy.amax(dim=(2, 3)) > 0), 'No positive values in the logit output'
    assert torch.any(yy.amin(dim=(2, 3)) < 0), 'No negative values in the logit output'


@torch.no_grad()
def test_output_shape(model):
    """ The shape of the outputs should match the shape of the input. """

    xx0 = torch.ones(2, 4, 255, 257)
    xx1 = torch.ones(3, 4, 400, 403)

    yy0 = model(xx0)
    yy1 = model(xx1)

    assert yy0.shape == (2, 10, 255, 257), 'Shape of the output does not match shape of the input'
    assert yy1.shape == (3, 10, 400, 403), 'Shape of the output does not match shape of the input'


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available.')
def test_switch_devices(model):
    """ There should be no problems with device switching. """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()

    xx = torch.ones(2, 4, 255, 257)

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
