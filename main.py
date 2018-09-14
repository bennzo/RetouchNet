import argparse
import setproctitle
import os
from utils import setup_cuda


def setup_main():
    opt = parse_args()
    procname = os.path.basename(opt.checkpoint_dir)
    setproctitle.setproctitle('retouchNet_{}'.format(procname))

    opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.name)
    print(f'Preparing summary and checkpoint directory {opt.checkpoint_dir}')
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs('%s/images/%s' % (opt.checkpoint_dir, opt.name), exist_ok=True)
    opt = setup_cuda(opt)

    print(opt)
    return opt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
    parser.add_argument('--train-data', type=str, required=True, help='Train data filelist.txt path')
    parser.add_argument('--test-data', type=str, required=True, help='Test data filelist.txt path')
    parser.add_argument('--checkpoint-dir', type=str, default='log', help='log dir')
    parser.add_argument('--name', type=str, default='retouchnet', help='name of the dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--loss_ratio', type=float, default=100,
                        help='ratio between patchgan and pixelwise losses. loss = loss_ratio*pixelwise + patchgan')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_height', type=int, default=480, help='size of image height')
    parser.add_argument('--img_width', type=int, default=720, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
    parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')

    # loading pretrained nets
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--benchmark', action='store_true', help="enables benchmarking")
    parser.add_argument('--pw_guide', action='store_true', help="uses a pointwise guide instead of an original curve one")

    # GPU args
    parser.add_argument('--no-cuda', action='store_true', help='disables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu_id', default=0, help='gpu ids: e.g. 0  0 1 2  0,2. use -1 for CPU')

    # Data pipeline and data augmentation
    data_grp = parser.add_argument_group('data pipeline')
    data_grp.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    data_grp.add_argument('--batch_size', default=30, type=int, help='size of a batch for each gradient update.')
    data_grp.add_argument('--rotate', action="store_true", help='rotate data augmentation.')
    data_grp.add_argument('--flipud', action="store_true", help='flip up/down data augmentation.')
    data_grp.add_argument('--fliplr', action="store_true", help='flip left/right data augmentation.')
    data_grp.add_argument('--random_crop', action="store_true", help='random crop data augmentation.')
    return parser.parse_args()