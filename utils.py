import os
import random
import numpy as np
import torch
from torch.backends import cudnn as cudnn


class ModelSaver:
    def __init__(self, path):
        self.best = float('inf')
        self.path = path
        self.epoch = 0

    def save_if_best(self, model, loss):
        self.epoch += 1
        os.makedirs(os.path.join(self.path, model.__class__.__name__), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.path, model.__class__.__name__, 'last_model.checkpoint'))
        if loss < self.best:
            self.best = loss
            torch.save(model.state_dict(), os.path.join(self.path, model.__class__.__name__, 'best_model.checkpoint'))


def load_or_init_models(m, opt):
    if opt.netG != '' and type(m).__name__ == 'RetouchGenerator':
        m.load_state_dict(torch.load(opt.netG))
    elif opt.netD != '' and type(m).__name__ == 'Discriminator':
        m.load_state_dict(torch.load(opt.netD))
    else:
        m.apply(weights_init_normal)
    return m


def update_stats(stats, measurments):
    for k, v in measurments.items():
        stats[k] += v
    return stats


def seed_all(opt):
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


def to_variables(tensors, cuda=None, device=None, test=False, **kwargs):
    if cuda is None:
        cuda = torch.cuda.is_available()

    variables = []
    for i, t in enumerate(tensors):
        variables.append(tensors[i].to(device))
        if test:
            variables[-1].requires_grad = False
    return variables


def setup_cuda(opt):
    if torch.cuda.is_available() and opt.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    opt.cuda = torch.cuda.is_available() and not opt.no_cuda

    if opt.cuda:
        print("WARNING: make sure you prepend \"CUDA_VISIBLE_DEVICES=<gpu_id>\" to your script to run on specified gpu")
        opt.device = torch.device(f'cuda:{opt.gpu_id}')
        torch.cuda.set_device(int(opt.gpu_id))
        torch.cuda.manual_seed_all(opt.manualSeed)
        print_cuda()
    else:
        opt.device = torch.device('cpu')
        print('Active CUDA Device: CPU')

    cudnn.benchmark = True
    return opt


def print_cuda():
    import torch
    import sys
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
