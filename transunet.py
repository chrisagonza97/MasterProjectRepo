import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms

class TransUNet(nn.Module):
    def __init__(self, in_channels=3, classes=3):
        super().__init__()
        self.encoder = TransUNetEncoder()
        self.decoder = TransUNetDecoder()

    def forward(self,x):
        encoder_output, skip_connections = self.encoder(x)
        decoder_output = self.decoder(encoder_output, skip_connections)
        return decoder_output
        

class TransformEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        encoders = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoders, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class TransUNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.cnn_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        layers = list(self.cnn_encoder.children())
        #print(list(self.cnn_encoder.children()))   #CNN + Transformer
        self.skip1 = nn.Sequential(*layers[:3])
        self.maxpool = layers[3]
        self.skip2 = layers[4] #1/4 resolution
        self.skip3 = layers[5] #1/8 resolution
        self.transformer_input = layers[6] #used for transformer 1/16 resolution
        self.linear_projection = nn.Linear(1024, 768)
        self.tranformers = TransformEncoder(embed_dim=768, num_heads=12, num_layers=12)
        
        # ADD POSITIONAL EMBEDDINGS
        #self.position_embeddings = nn.Parameter(torch.randn(1, 256, 768))  # (1, seq_len, embed_dim)


        #self.decoder   
    def forward(self, x):
        """ Extract CNN feature maps before passing to Transformer. """
        skip1 = self.skip1(x)  # 1/2 resolution (before pooling)
        x = self.maxpool(skip1)  # Apply MaxPool separately
        #print("maxpool shape")
        #print(x.shape) 
        skip2 = self.skip2(x)  # 1/4 resolution
        skip3 = self.skip3(skip2)  # 1/8 resolution
        transformer_input = self.transformer_input(skip3)  # 1/16 resolution
        #print shape
        #print("transformer_input shape")
        #print(transformer_input.shape ) #should be (batch_size, 1024, 16, 16)

        #print("skip 1 shape")
        #print(skip1.shape) #should be (batch_size, 64, 64, 64)
        #print("skip 2 shape")
        #print(skip2.shape) #should be (batch_size, 256, 32, 32)
        #print("skip 3 shape")
        #print(skip3.shape) #should be (batch_size, 512, 16, 16)

        #next do linear projection to get the input for the transformer
        # Reshape to match Transformer input: (batch_size, H*W, channels)
        batch_size, channels, H, W = transformer_input.shape
        #transformer_input = transformer_input.permute(0, 2, 3, 1).reshape(batch_size, H * W, channels)
        transformer_input = transformer_input.flatten(2).transpose(1, 2)

        #(batch_size, seq_len, embed_dim)
        transformer_input = self.linear_projection(transformer_input)

        # ADD POSITION EMBEDDINGS
        #transformer_input += self.position_embeddings[:, :transformer_input.shape[1], :]  

        transformer_output = self.tranformers(transformer_input)

        encoder_output = transformer_output.permute(0, 2, 1).reshape(batch_size, 768, H, W)

        return encoder_output, [skip1, skip2, skip3]  # Return for TransUNet
    
class TransUNetDecoder(nn.Module):
    def __init__(self, in_channels=768, skip_channels=[512, 256, 64]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(512 + skip_channels[0], 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        #print(self.conv1)
        self.conv2 = nn.Sequential(nn.Conv2d(256 + skip_channels[1], 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128 + skip_channels[2], 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x, skip_connections):

        #x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.conv0(x)
        #print(x.shape)
        x = self.upsample(x)
        #print(x.shape)
        x = torch.cat((x, skip_connections[2]), dim=1)
        #print(x.shape)
        x = self.conv1(x)
        x = self.upsample(x)

        x = torch.cat((x, skip_connections[1]), dim=1)
        x = self.conv2(x)
        x = self.upsample(x)

        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.final_conv(x)

        #x = torch.softmax(x, dim=1)

        return x

    

