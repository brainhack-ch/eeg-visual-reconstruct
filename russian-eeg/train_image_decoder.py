import torch
from torch import optim
from image_decoder import ImageDecoder

from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F

from siamese_sampler import SiameseSampler

import time
import logger
import tensorboard_logger as tb


logger = logger.getLogger('train')


if torch.cuda.is_available():
    device = torch.device("cuda")
    pin_memory = True
else:
    device = torch.device("cpu")
    pin_memory = False
print("Device: {}".format(device))


def get_grad_sum(model):
    grads = {}
    # nans = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.sum()
        # else:
        #     nans += 1
    return grads  #, nans


def train_image_decoder(args):
    print("Start training")
    image_decoder = ImageDecoder()
    image_decoder.to(device)
    optimizer = optim.Adam(image_decoder.parameters(), args.lr)

    transform = transforms.Compose([
        transforms.RandomRotation(5, resample=False),
        transforms.Resize(250, interpolation=3),
        transforms.CenterCrop(220),
        transforms.RandomCrop(192),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    # TODO: Random crop or reshape to 192 x 192
    dataset = datasets.ImageFolder(
        root=args.data_folder,
        transform=transform)

    dataloader = data.DataLoader(
        dataset,
        batch_sampler=SiameseSampler(dataset, args.batch_size, args.n_batches))

    start_time = time.time()

    loss_min = 1000000

    for e in range(args.n_epochs):
        image_decoder.train()
        for b, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # labels = batch[-1]
            # labels_1 = labels[:args.batch_size]
            # labels_2 = labels[args.batch_size:]

            batch = torch.cat(batch[:-1], 0)
            input_1 = batch[:args.batch_size]
            input_2 = batch[args.batch_size:]
            input_1, input_2 = input_1.to(device), input_2.to(device)

            latent_1, output_1 = image_decoder(input_1)
            latent_2, output_2 = image_decoder(input_2)

            # TODO: Calculate the batch vector, then add and average
            # TODO: Check the input size for VGG!!!
            # reconstruction_loss = (
            #     ((output_1 - input_1)**2).sum(1).sqrt() +
            #     ((output_2 - input_2)**2).sum(1).sqrt()
            # ) / (192 * 192 * args.batch_size)
            # reconstruction_loss = reconstruction_loss.mean()
            reconstruction_loss = \
                F.mse_loss(input_1, output_1) + \
                F.mse_loss(input_2, output_2)

            # d = ((latent_2 - latent_1)**2).sum(1).sqrt() / 20
            # l_const = (labels_1 == labels_2).float()
            # t = l_const - torch.exp(args.eta * d)
            # t = t.detach()

            # distance_loss = t * d + (1 - t) * \
            #     torch.sigmoid(1 - d)**2
            # distance_loss = distance_loss.mean()

            distance_loss = F.mse_loss(latent_1, latent_2)
            distance_loss = distance_loss.mean()

            # modulus = (latent_1**2).sum(1).sqrt() * \
            #     (latent_2**2).sum(1).sqrt()
            # angle_loss = (1 - t) * (latent_1 * latent_2).sum(1) / modulus
            # angle_loss = angle_loss.mean()

            angle_loss = F.cosine_similarity(latent_1, latent_2)
            angle_loss = angle_loss.mean()

            loss = 2 * distance_loss + 2 * angle_loss + reconstruction_loss

            # loss = reconstruction_loss

            loss.backward()

            print("Grad")
            print("".join([f"{k}: {v}\n" for k, v in get_grad_sum(image_decoder).items()]))
            optimizer.step()

            passed_time = time.time() - start_time

            if b % args.log_batches == 0:
                # Printing and logging
                logger.info(
                    "Ep {:3d} b {:4d}: time {}, loss {:.6f}".format(
                        e, b,
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(passed_time)),
                        loss
                    )
                )
                tb.log_value('loss_time', loss, int(passed_time))
                tb.log_value(
                    'loss_rec_time', reconstruction_loss, int(passed_time))
                tb.log_value(
                    'loss_dist_time', distance_loss, int(passed_time))
                tb.log_value(
                    'loss_angle_time', angle_loss, int(passed_time))

        loss_train = loss

        tb.log_value('loss_e', loss, e)
        tb.log_value(
            'loss_rec_e', reconstruction_loss, e)
        # tb.log_value(
        #     'loss_dist_e', distance_loss, e)
        # tb.log_value(
        #     'loss_angle_e', angle_loss, e)

        image_decoder.eval()

        for batch in dataloader:
            # labels = batch[-1]
            # labels_1 = labels[:args.batch_size]
            # labels_2 = labels[args.batch_size:]

            batch = torch.cat(batch[:-1], 0)
            input_1 = batch[:args.batch_size]
            input_2 = batch[args.batch_size:]
            input_1, input_2 = input_1.to(device), input_2.to(device)

            latent_1, output_1 = image_decoder(input_1)
            latent_2, output_2 = image_decoder(input_2)

            # reconstruction_loss = (
            #     ((output_1 - input_1)**2).sum(1).sqrt() +
            #     ((output_2 - input_2)**2).sum(1).sqrt()
            # ) / (192 * 192 * args.batch_size)
            # reconstruction_loss = reconstruction_loss.mean()
            reconstruction_loss = \
                F.mse_loss(input_1, output_1) + \
                F.mse_loss(input_2, output_2)
            reconstruction_loss = reconstruction_loss.mean()

            # d = ((latent_2 - latent_1)**2).sum(1).sqrt() / 20
            # l_const = (labels_1 == labels_2).float()
            # t = l_const - torch.exp(args.eta * d)
            # t = t.detach()

            # distance_loss = t * d + (1 - t) * \
            #     torch.sigmoid(1 - d)**2
            # distance_loss = distance_loss.mean()

            distance_loss = F.mse_loss(latent_1, latent_2)
            distance_loss = distance_loss.mean()

            # modulus = (latent_1**2).sum(1).sqrt() * \
            #     (latent_2**2).sum(1).sqrt()
            # angle_loss = (1 - t) * (latent_1 * latent_2).sum(1) / modulus
            # angle_loss = angle_loss.mean()

            angle_loss = F.cosine_similarity(latent_1, latent_2)
            angle_loss = angle_loss.mean()

            loss = 2 * distance_loss + 2 * angle_loss + reconstruction_loss

            # loss = reconstruction_loss

            tb.log_value('loss_e_val', loss, e)
            tb.log_value(
                'loss_rec_e_val', reconstruction_loss, e)
            tb.log_value(
                'loss_dist_e_val', distance_loss, e)
            tb.log_value(
                'loss_angle_e_val', angle_loss, e)
            break

        if loss < loss_min:
            torch.save(
                image_decoder.state_dict(),
                args.sum_base_dir + '/model_best.pth')
            torch.save(
                optimizer.state_dict(),
                args.sum_base_dir + '/optimizer_best.pth')
            logger.info("Next best, saved the model")

        logger.info("Epoch done: loss_train {:.6f}, loss_val: {:.6f}".format(
            loss_train, loss))
