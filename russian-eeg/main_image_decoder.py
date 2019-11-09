import torch
import argparse

from train_image_decoder import train_image_decoder

import os
import time
import logger
import tensorboard_logger as tb


logger = logger.getLogger('main')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate (default: 0.00005)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--data-folder', type=str, default='data/_screenshots',
                    help='data folder')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size')
parser.add_argument('--n-batches', type=int, default=100,
                    help='number of batches per epoch')
parser.add_argument('--log-batches', type=int, default=1,
                    help='log every _ batches')
parser.add_argument('--eta', type=int, default=1e-3,
                    help='eta from the paper')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--short-description', default='no_descr',
                    help='short description of the run params '
                    '(used in TensorBoard)')


def setup_loggings(args):
    current_path = os.path.dirname(os.path.realpath(__file__))
    args.sum_base_dir = (current_path + '/runs/image_decoder/{}({})').format(
        time.strftime('%Y.%m.%d-%H.%M'), args.short_description)

    args_list = [f'{k}: {v}\n' for k, v in vars(args).items()]
    logger.info("\nArguments:\n----------\n" + ''.join(args_list))
    logger.info('Logging run logs to {}'.format(args.sum_base_dir))
    tb.configure(args.sum_base_dir)


if __name__ == '__main__':

    args = parser.parse_args()
    setup_loggings(args)

    torch.manual_seed(args.seed)

    train_image_decoder(args)
