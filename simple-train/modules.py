from torch import nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBnRelu, self).__init__()
        self.convBnRelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.convBnRelu(x)
        return x
    

class ConvBnPRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBnPRelu, self).__init__()
        self.convBnPRelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(num_parameters=out_channels)
        )
    def forward(self, x):
        x = self.convBnPRelu(x)
        return x



class SimpleModel(nn.Module):
    def __init__(self, in_channels=3, out_channels = 1024, out_features=100):
        super(SimpleModel, self).__init__()
        self.backbone = nn.Sequential(
            ConvBnPRelu(in_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(256, 512),
            nn.Conv2d(512, out_channels, kernel_size=1),
        )
        self.classfier = nn.Sequential(
            nn.Linear(out_channels*2*2, 512),   # 32 * 32 四次pooling -> 2*2
            nn.Linear(512, 386),
            nn.Linear(386, out_features)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x
    
class ComplexModel(nn.Module):
    def __init__(self, in_channels=3, out_channels = 1024, out_features=100):
        super(ComplexModel, self).__init__()
        self.backbone = nn.Sequential(
            ConvBnPRelu(in_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnPRelu(256, 384),
            ConvBnPRelu(384, 640),
            ConvBnPRelu(640, 768),
            nn.Conv2d(768, out_channels, kernel_size=1),
        )
        self.classfier = nn.Sequential(
            nn.Linear(out_channels*2*2, 512),   # 32 * 32 四次pooling -> 2*2
            nn.Linear(512, 386),
            nn.Linear(386, out_features)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x