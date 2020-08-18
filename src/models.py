import torch
import torch.nn as nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


class MetaLayer(nn.Module):
    def __init__(self, input_features=10, num_features=256, p=0.2):
        super(MetaLayer, self).__init__()
        self.meta = nn.Sequential(nn.Linear(input_features, num_features),
                                  nn.BatchNorm1d(num_features),
                                  nn.ReLU(),
                                  nn.Dropout(p=p),
                                  nn.Linear(num_features, num_features//2),
                                  nn.BatchNorm1d(num_features//2),
                                  nn.ReLU(),
                                  nn.Dropout(p=p/2))

    def forward(self, x):
        return self.meta(x)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class EfficientNets(nn.Module):
    def __init__(self, model_name='efficientnet-b0',
                 num_class=1,
                 pretrained=True,
                 num_meta=None,
                 num_features=256):
        super(EfficientNets, self).__init__()
        self.num_meta = num_meta
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name)
        if num_meta is not None:
            self.model._fc = nn.Linear(
                self.model._fc.in_features, num_features)
            self.meta = MetaLayer(num_meta, num_features)
            self.last_layer = nn.Linear(
                num_features+(num_features//2), num_class)
        else:
            self.model._fc = nn.Linear(self.model._fc.in_features, num_class)

    def forward(self, x):
        if self.num_meta is not None:
            x, meta_features = x
            cnn_features = self.model(x)
            meta_features = self.meta(meta_features)
            features = torch.cat((cnn_features, meta_features), dim=1)
            output = self.last_layer(features)
            return output
        else:
            output = self.model(x)
            return output


class PretrainedNets(nn.Module):
    def __init__(self, model_name='resnet18',
                 num_class=1,
                 pretrained=True,
                 num_meta=None):
        super(PretrainedNets, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__[
                model_name](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__[model_name](pretrained=None)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(self.model.last_linear.in_features, num_class)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = self.pooling(x).reshape(bs, -1)
        output = self.output(x)
        return output
