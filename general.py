import argparse
import logging
import random

import numpy as np
import os
import setproctitle
import torch
import torch.backends.cudnn as cudnn

log = logging.getLogger("root")
log.setLevel(logging.INFO)


def setup_main():
    opt = parse_args()
    procname = os.path.basename(opt.checkpoint_dir)
    setproctitle.setproctitle('retouchNet_{}'.format(procname))

    opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.name)
    log.info('Preparing summary and checkpoint directory {}'.format(opt.checkpoint_dir))
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    setup_cuda(opt)

    print(opt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True,
                        help='input directory containing the training .tfrecords or images.')
    parser.add_argument('--test-data', required=True, type=str, help='directory with the validation data.')
    parser.add_argument('--checkpoint_dir', default='log', help='folder to output images and model checkpoints')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    # parser.add_argument('--nci', type=int,
    #                     default=3, help='size of the input channels')
    parser.add_argument('--nco', type=int,
                        default=3, help='size of the output channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
    parser.add_argument('--niter', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')

    # GPU args
    parser.add_argument('--no-cuda', action='store_true', help='disables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0],
                        help='gpu ids: e.g. 0  0 1 2  0,2. use -1 for CPU')

    # Data pipeline and data augmentation
    data_grp = parser.add_argument_group('data pipeline')
    data_grp.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    data_grp.add_argument('--batch_size', default=16, type=int, help='size of a batch for each gradient update.')
    data_grp.add_argument('--rotate', action="store_true", help='rotate data augmentation.')
    data_grp.add_argument('--flipud', action="store_true", help='flip up/down data augmentation.')
    data_grp.add_argument('--fliplr', action="store_true", help='flip left/right data augmentation.')
    data_grp.add_argument('--random_crop', action="store_true", help='random crop data augmentation.')

    return parser.parse_args()


def setup_cuda(opt):
    if torch.cuda.is_available() and not opt.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    opt.cuda = torch.cuda.is_available() and not opt.no_cuda

    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True


def seed_all(opt):
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


def to_variables(*tensors, cuda=None, test=False, **kwargs):
    if cuda is None:
        cuda = torch.cuda.is_available()

    for t in tensors:
        if cuda:
            t.cuda()
        if test:
            t.requires_grad = False
    return tensors
