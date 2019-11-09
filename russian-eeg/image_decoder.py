import torch
import torchvision.models as models
from torch import nn


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.encoder = models.vgg11(pretrained=True).features
        # for p in self.encoder.parameters():
        #     p.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # for p in self.avgpool.parameters():
        #     p.requires_grad = False

        encoder_output_size = 512 * 7 * 7
        fc_sizes = [20, 128 * 12 * 12]
        dc_channels = [256, 128, 64, 32, 3]
        upsample_sizes = [12, 24, 48, 96, 192]
        self.fc_sizes = fc_sizes

        self.to_latent = nn.Sequential(
            nn.Linear(encoder_output_size, fc_sizes[0]),
            nn.ReLU()
        )

        self.from_latent = nn.Sequential(
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU()
        )

        # self.to_latent = nn.Sequential(
        #     nn.Conv2d(self.encoder_output_size, fc_sizes[0], kernel_size=1),
        #     nn.ReLU()
        # )

        # self.from_latent = nn.Sequential(
        #     nn.Conv2d(fc_sizes[0], fc_sizes[1], kernel_size=1),
        #     nn.ReLU()
        # )

        self.decoder = nn.Sequential(
            nn.Upsample(size=(upsample_sizes[0], upsample_sizes[0])),
            nn.ConvTranspose2d(
                in_channels=fc_sizes[1], out_channels=dc_channels[0],
                kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(size=(upsample_sizes[1], upsample_sizes[1])),
            nn.ConvTranspose2d(
                in_channels=dc_channels[0], out_channels=dc_channels[1],
                kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(size=(upsample_sizes[2], upsample_sizes[2])),
            nn.ConvTranspose2d(
                in_channels=dc_channels[1], out_channels=dc_channels[2],
                kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(size=(upsample_sizes[3], upsample_sizes[3])),
            nn.ConvTranspose2d(
                in_channels=dc_channels[2], out_channels=dc_channels[3],
                kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(size=(upsample_sizes[4], upsample_sizes[4])),
            nn.ConvTranspose2d(
                in_channels=dc_channels[3], out_channels=dc_channels[4],
                kernel_size=3, padding=1),
            nn.Tanh()
        )

        # for layer in self.encoder:
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.xavier_uniform(layer.weight)

        print(self.to_latent[0])
        nn.init.xavier_uniform_(self.to_latent[0].weight)
        nn.init.xavier_uniform_(self.from_latent[0].weight)

        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                print(layer)
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        #x = torch.flatten(x, 1).reshape(-1, self.encoder_output_size, 1, 1)
        x = torch.flatten(x, 1)
        latent = self.to_latent(x)
        # latent_out = latent.flatten(1)
        # print(f"to_latent {latent.size()}")
        x = self.from_latent(latent)
        # print(f"from_latent {x.size()}")
        x = x.view(-1, self.fc_sizes[-1], 1, 1)
        # print(f"reshape {x.size()}")
        x = self.decoder(x)
        # print(f"decoder {x.size()}")
        return latent, x
    pass
