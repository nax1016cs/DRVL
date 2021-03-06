import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class ResNeXt(nn.Module):
    def __init__(self, output, pretrained=True):
        super(ResNeXt, self).__init__()
        pretrained_model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext101_32x8d',
            pretrained=True)
        self.classify = nn.Linear(
            pretrained_model._modules['fc'].in_features,
            output)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


if __name__ == '__main__':
    model = ResNeXt(output=200)
