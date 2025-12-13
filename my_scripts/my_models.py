import torch
import torch.nn as nn

# Define a Convolution Block:
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

# # define a silly linear block:
# def linear_layer(in_size):
#     return nn.Sequential(
#         nn.Linear(in_size, in_size/2),
#         nn.ReLU(inplace=True)
#     )

class LargeCNN(nn.Module):
    """Small CNN with 3 convolutional blocks"""
    def __init__(self, in_channels=3, num_classes=2, channels=(16, 32, 64,128,128,256), dropout=0.2):
        super().__init__()
        c1, c2, c3,c4,c5,c6 = channels

        # convolutional feature extractor
        self.features = nn.Sequential(
            conv_block(in_channels, c1),
            conv_block(c1, c2),
            conv_block(c2, c3),
            conv_block(c3, c4),
            conv_block(c4, c5),
            conv_block(c5, c6),
            nn.Dropout(dropout)
        )

        # global average pooling (nn.AdaptiveAvgPool2d) compresses the spatial dimensions to NxCx1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # linear layer for classification
        self.classifier = nn.Linear(c6, num_classes)

    #  forward pass (how the input passes through the CNN layers)
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        # flatten until dimension 1 so that NxCx1x1 becomes NxC- suitable for linear layer
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        return x
# nn.Module is the parent class for neural network modules in PyTorch
class SmallCNN(nn.Module):
    """Small CNN with 3 convolutional blocks"""
    def __init__(self, in_channels=3, num_classes=2, channels=(16, 32, 64), dropout=0.2):
        super().__init__()
        c1, c2, c3 = channels
        
        # convolutional feature extractor
        self.features = nn.Sequential(
            conv_block(in_channels, c1),
            conv_block(c1, c2),
            conv_block(c2, c3),
            nn.Dropout(dropout) 
        )
        
        # global average pooling (nn.AdaptiveAvgPool2d) compresses the spatial dimensions to NxCx1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # linear layer for classification
        self.classifier = nn.Linear(c3, num_classes)
    
    #  forward pass (how the input passes through the CNN layers)
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        # flatten until dimension 1 so that NxCx1x1 becomes NxC- suitable for linear layer
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        return x
    
    # nn.Module is the parent class for neural network modules in PyTorch
class SmallMLP(nn.Module):
    """Small silly naive MLP baseline"""
    def __init__(self, in_channels=3, num_classes=2, size=((96, 96)), dropout=0.2):
        super().__init__()
        # c1, c2, c3 = channels
        in_features = in_channels * size[0] * size[1]

        # similar number of parameters as the CNN we're using
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
            
    #  forward pass (how the input passes through the layers)
    def forward(self, x):
        # x = self.features(x)
        # x = self.gap(x)
        # # flatten until dimension 1 so that NxCx1x1 becomes NxC- suitable for linear layer
        # x = torch.flatten(x, 1)
        # # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
