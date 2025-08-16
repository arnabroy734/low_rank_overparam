from torch import nn
import yaml
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=1, padding='same', kernel_size=kernel)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.activation(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return x

class ClsBlock(nn.Module):
    def __init__(self, in_dim: int, categories: int):
        super().__init__()
        self.proj = nn.Linear(in_features=in_dim, out_features=categories)
    def forward(self, x):
        x = self.proj(x)
        return x

class BaseModel(nn.Module):
    def __init__(
        self,
        conv_blocks: int,
        channels: list[int],
        kernels: list[int],
        fc_blocks: int,
        fc_dims: list[int],
        categories: int
    ):
        super().__init__()
        self.convblocks = nn.ModuleDict()
        for i in range(conv_blocks):
            self.convblocks[f'conv{i}'] = ConvBlock(in_channel=channels[i], out_channel=channels[i+1], kernel=kernels[i])
        self.convblocks = nn.Sequential(*self.convblocks.values())

        self.pool = nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()
        
        self.fcblocks = nn.ModuleDict()
        for i in range(fc_blocks):
            self.fcblocks[f'fcblock{i}'] = FCBlock(in_dim=fc_dims[i], out_dim=fc_dims[i+1])
        self.fcblocks = nn.Sequential(*self.fcblocks.values())
        
        self.classifier = ClsBlock(in_dim=fc_dims[-1], categories=categories)

    def forward(self, x):
        x = self.convblocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fcblocks(x)
        x = self.classifier(x)
        return x

def create_basemodel(configpath='config.yaml'):
    try:
        with open(configpath, "r") as f:
            config = yaml.safe_load(f)
            f.close()
        torch.manual_seed(42)
        basemodel = BaseModel(
            conv_blocks=config['model_config']['conv_blocks'],
            channels=config['model_config']['channels'],
            kernels=config['model_config']['kernels'],
            fc_blocks=config['model_config']['fc_blocks'],
            fc_dims=config['model_config']['fc_dims'],
            categories=config['model_config']['categories']
        )
        return basemodel


    except FileNotFoundError as e:
        raise FileNotFoundError(e)
    except KeyError as e:
        raise KeyError(e)
    except Exception as e:
        raise Exception(f"Basemodel could not be created {e}")
