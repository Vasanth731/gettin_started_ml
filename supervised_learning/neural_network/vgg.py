import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# defining the layer as an array
VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        
        # creating the conv layers
        self.conv_layers = self.create_conv_layers(VGG16)
        
        # creating fully connected layers
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1) # batchsize * all width,height multiplied
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=(3,3), 
                        stride=(1,1), 
                        padding=(1,1)
                    ),
                    nn.BatchNorm2d(x), 
                    nn.ReLU(),
                ]
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]

        return nn.Sequential(*layers) # *layers = unpacks all the layers
    
model = VGG_net(in_channels=3, num_classes=14)
# x = torch.randn(1,3,224,224) # (batch_size,color_channel,width,height)
# print(model(x).shape)
print(model)    

