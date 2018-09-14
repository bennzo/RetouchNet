import torch
import torch.nn as nn


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class PatchLoss(nn.Module):
    def __init__(self):
        # Loss functions
        self.criterion_GAN = nn.MSELoss()

    def forward(self, pred_real, pred_fake):
        # Real loss
        ones = torch.ones(pred_real.size(), requires_grad=False)
        loss_real = self.criterion_GAN(pred_real, ones)

        # Fake loss
        zeros = torch.zeros(pred_fake.size(), requires_grad=False)
        loss_fake = self.criterion_GAN(pred_fake, zeros)

        # Total loss
        return 0.5 * (loss_real + loss_fake)
