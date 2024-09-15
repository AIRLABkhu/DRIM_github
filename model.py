import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


from saic_depth_completion.modeling.blocks import AdaptiveBlock, FusionBlock, CRPBlock, SharedEncoder
from saic_depth_completion.ops.spade import SPADE


class DRIM(nn.Module):
    def __init__(self, *, pretrained=True, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = DRIM_shared_encoder(pretrained=True, encoder=models.resnet50, in_channels=1, out_channels=1)
        self.mask_decoder = DRIM_mask_decoder()
        self.depth_restoration_decoder = DRIM_depth_restoration_decoder()


    def forward(self, x):
        
        block = self.encoder(x)

        mask_pred = self.mask_decoder(block)

        probs = torch.softmax(mask_pred, dim=1)
        mask = torch.argmax(probs, dim=1).unsqueeze(1)

        sensor_inter_mask = torch.zeros_like(mask).float()
        sensor_inter_mask[mask==1] = 1
        sensor_inter_mask[mask==2] = -1

        x = self.depth_restoration_decoder(block, sensor_inter_mask.detach())

        return x, mask_pred



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


class DRIM_mask_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.mask_up_conv6 = up_conv(2048, 512)
        self.mask_conv6 = double_conv(512 + 1024, 512)
        self.mask_up_conv7 = up_conv(512, 256)
        self.mask_conv7 = double_conv(256 + 512, 256)
        self.mask_up_conv8 = up_conv(256, 128)
        self.mask_conv8 = double_conv(128 + 256, 128)
        self.mask_up_conv9 = up_conv(128, 64)
        self.mask_conv9 = double_conv(64 + 64, 64)
        self.mask_up_conv10 = up_conv(64, 32)


        self.mask_conv = nn.Conv2d(32, 3, kernel_size=1) # Binary classification


    def forward(self, block):
        
        x = self.mask_up_conv6(block[4])
        x = torch.cat([x, block[3]], dim=1)
        x = self.mask_conv6(x)

        x = self.mask_up_conv7(x)
        x = torch.cat([x, block[2]], dim=1)
        x = self.mask_conv7(x)

        x = self.mask_up_conv8(x)
        x = torch.cat([x, block[1]], dim=1)
        x = self.mask_conv8(x)

        x = self.mask_up_conv9(x)
        x = torch.cat([x, block[0]], dim=1)
        x = self.mask_conv9(x)

        x = self.mask_up_conv10(x)

        # Outputs
        mask_pred = self.mask_conv(x)

        return mask_pred


class DRIM_shared_encoder(nn.Module):
    def __init__(self, *, pretrained=True, encoder=models.resnet50, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = encoder(pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return [block1, block2, block3, block4, block5]


class DRIM_depth_restoration_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.predict_log_depth      = True
        self.activation             = nn.LeakyReLU(0.2,inplace = True)
        self.modulation             = SPADE
        self.channels               = 256
        self.upsample               = "bilinear"
        self.use_crp                = True
        self.mask_encoder_ksize     = 3

        self.modulation32 = AdaptiveBlock(
            self.channels, self.channels, self.channels,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation16 = AdaptiveBlock(
            self.channels // 2, self.channels // 2, self.channels // 2,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation8  = AdaptiveBlock(
            self.channels // 4, self.channels // 4, self.channels // 4,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation4  = AdaptiveBlock(
            self.channels // 8, self.channels // 8, self.channels // 8,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )

        self.modulation4_1 = AdaptiveBlock(
            self.channels // 8, self.channels // 16, self.channels // 8,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation4_2 = AdaptiveBlock(
            self.channels // 16, self.channels // 16, self.channels // 16,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )

        self.mask_encoder = SharedEncoder(
            out_channels=(
                self.channels, self.channels // 2, self.channels // 4,
                self.channels // 8, self.channels // 8, self.channels // 16
            ),
            scales=(32, 16, 8, 4, 2, 1),
            upsample=self.upsample,
            activation=self.activation,
            kernel_size=self.mask_encoder_ksize
        )


        self.fusion_32x16 = FusionBlock(self.channels // 2, self.channels, upsample=self.upsample)
        self.fusion_16x8  = FusionBlock(self.channels // 4, self.channels // 2, upsample=self.upsample)
        self.fusion_8x4   = FusionBlock(self.channels // 8, self.channels // 4, upsample=self.upsample)

        self.adapt1 = nn.Conv2d(1024, self.channels, 1, bias=False)
        self.adapt2 = nn.Conv2d(512, self.channels // 2, 1, bias=False)
        self.adapt3 = nn.Conv2d(256, self.channels // 4, 1, bias=False)
        self.adapt4 = nn.Conv2d(64, self.channels // 8, 1, bias=False)


        self.crp1 = CRPBlock(self.channels, self.channels)
        self.crp2 = CRPBlock(self.channels // 2, self.channels // 2)
        self.crp3 = CRPBlock(self.channels // 4, self.channels // 4)
        self.crp4 = CRPBlock(self.channels // 8, self.channels // 8)

        self.predictor = nn.Sequential(*[
            nn.Conv2d(self.channels // 16, self.channels // 16, 1, padding=0, groups=self.channels // 16),
            nn.Conv2d(self.channels // 16, 1, 3, padding=1)
        ])


    def forward(self, block, sensor_inter_mask):
        
        f1 = self.crp1(self.adapt1(block[3]))
        f2 = self.adapt2(block[2])
        f3 = self.adapt3(block[1])
        f4 = self.adapt4(block[0])

        mask_features = self.mask_encoder(sensor_inter_mask)
        
        x = self.modulation32(f1, mask_features[0])
        x = self.fusion_32x16(f2, x)
        x = self.crp2(x)

        x = self.modulation16(x, mask_features[1])
        x = self.fusion_16x8(f3, x)
        x = self.crp3(x)

        x = self.modulation8(x, mask_features[2])
        x = self.fusion_8x4(f4, x)
        x = self.crp4(x)

        x = self.modulation4(x, mask_features[3])

        x = F.interpolate(x, scale_factor=2, mode=self.upsample)
        x = self.modulation4_1(x, mask_features[4])
        #x = F.interpolate(x, scale_factor=2, mode=self.upsample)
        x = self.modulation4_2(x, mask_features[5])

        return self.predictor(x)