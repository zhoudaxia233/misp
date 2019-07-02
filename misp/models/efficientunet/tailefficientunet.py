from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet

__all__ = ['EfficientUnet', 'get_tailefficientunet_b0', 'get_tailefficientunet_b1', 'get_tailefficientunet_b2',
           'get_tailefficientunet_b3', 'get_tailefficientunet_b4', 'get_tailefficientunet_b5',
           'get_tailefficientunet_b6', 'get_tailefficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = (output.size()[1], output)

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = (output.size()[1], output)

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.out_channels = out_channels
        self.concat_input = concat_input

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, (n_channels, x) = blocks.popitem()

        x = up_conv(n_channels, 512)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 512)(x)

        x = up_conv(512, 256)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 256)(x)

        x = up_conv(256, 128)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 128)(x)

        x = up_conv(128, 64)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 64)(x)

        if self.concat_input:
            x = up_conv(64, 32)(x)
            x = torch.cat([x, input_], dim=1)
            x = double_conv(x.size(1), 32)(x)

        custom_head_in_channels = x.size(1) * 2

        mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
        ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        x = torch.cat([mp, ap], dim=1)

        x = x.view(x.size(0), -1)

        x = custom_head(custom_head_in_channels, self.out_channels)(x)

        return x


def get_tailefficientunet_b0(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b1(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b2(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b3(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b4(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b5(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b6(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model


def get_tailefficientunet_b7(*, n_classes: int, concat_input: bool = True, pretrained: bool = True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=n_classes, concat_input=concat_input)
    return model
