import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )

def custom_head(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(),
        nn.Linear(512, out_channels)
    )

class TailUnet(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.bottleneck = double_conv(512, 1024)
        self.up_conv5 = up_conv(1024, 512)
        self.conv5 = double_conv(1024, 512)
        self.up_conv6 = up_conv(512, 256)
        self.conv6 = double_conv(512, 256)
        self.up_conv7 = up_conv(256, 128)
        self.conv7 = double_conv(256, 128)
        self.up_conv8 = up_conv(128, 64)
        self.conv8 = double_conv(128, 64)
        self.custom_head = custom_head(64 * 2, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self._weights_init()
    
    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.conv4(x)
        x = self.maxpool(conv4)

        bottleneck = self.bottleneck(x)

        x = self.up_conv5(bottleneck)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv5(x)

        x = self.up_conv6(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv8(x)
        
        mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
        ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        x = torch.cat([mp, ap], dim=1)
        
        x = x.view(x.size(0), -1)
        
        x = self.custom_head(x)

        return x
