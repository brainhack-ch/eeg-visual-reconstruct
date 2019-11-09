import torch
import numpy as np

from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from siamese_sampler import SiameseSampler
from image_decoder import ImageDecoder

from matplotlib import pyplot as plt


transform = transforms.Compose([
    transforms.Resize(192, interpolation=3),
    transforms.RandomCrop(192),
    transforms.ToTensor()
])

# TODO: Random crop or reshape to 192 x 192
dataset = datasets.ImageFolder(
    root='data/_screenshots',
    transform=transform)

dataloader = data.DataLoader(
    dataset,
    batch_sampler=SiameseSampler(dataset, 8, 1))

model = ImageDecoder()
model.load_state_dict(torch.load('runs/image_decoder/2019.11.09-13.27(no_descr)/model_best.pth', map_location='cpu'))

latents = []
labels = []

for b, batch in enumerate(dataloader):
    labels.append(batch[-1])
    print(batch[0].shape)
    latent, output = model(batch[0])

    input_ = batch[0].numpy()
    output = output.detach().numpy()

    latents.append(latent)

    for i in range(len(input_)):
        plt.imshow(np.moveaxis(input_[i], 0, -1))
        plt.axis('off')
        plt.savefig(f'figs_1/input_{b}_{i}.png', dpi=300)
        plt.close()

        plt.imshow(np.moveaxis(output[i], 0, -1))
        plt.axis('off')
        plt.savefig(f'figs_1/output_{b}_{i}.png', dpi=300)
        plt.close()

# latents = torch.cat(latents, 0).detach().numpy()
# labels = torch.cat(labels).detach().numpy()

# np.savetxt('latents_new.txt', latents)
# np.savetxt('labents_new.txt', labels)
