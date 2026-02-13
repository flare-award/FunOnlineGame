import torch

from autopilot_bot.model import TinyCNN


def test_model_forward_shape():
    model = TinyCNN(num_classes=7, in_channels=3)
    x = torch.randn(1, 3, 96, 96)
    out = model(x)
    assert out.shape == (1, 7)
