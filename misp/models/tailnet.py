import torch
import torch.nn as nn
from .tailunet import double_conv, custom_head

__all__ = ['tailnet_for_tailunet', 'tailnet_for_tailresunet', 'tailnet_for_tailvggunet',
           'tailnet_for_tailefficientunet_concat_input', 'tailnet_for_tailefficientunet_no_concat_input']


class TailNet(nn.Module):
    """Used for comparison with TailUnet.
    """
    def __init__(self, channels_before_pooling: int, n_classes: int):
        super().__init__()

        self.conv = double_conv(3, channels_before_pooling)
        self.custom_head = custom_head(channels_before_pooling * 2, n_classes)

        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)

        mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
        ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        x = torch.cat([mp, ap], dim=1)

        x = x.view(x.size(0), -1)

        x = self.custom_head(x)

        return x


def tailnet_for_tailunet(*, n_classes: int):
    return TailNet(64, n_classes)


def tailnet_for_tailresunet(*, n_classes: int):
    return TailNet(32, n_classes)


def tailnet_for_tailvggunet(*, n_classes: int):
    return TailNet(32, n_classes)


def tailnet_for_tailefficientunet_concat_input(*, n_classes: int):
    return TailNet(32, n_classes)


def tailnet_for_tailefficientunet_no_concat_input(*, n_classes: int):
    return TailNet(64, n_classes)
