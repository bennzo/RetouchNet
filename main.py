"""
import argparse
import logging

import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from data.data import FivekDataset

log = logging.getLogger("root")
log.setLevel(logging.INFO)

def run(args):
    train_data = DataLoader(FivekDataset(args.))
    model = LocalFeatureNet()

    # Add code for using CUDA here if it is available
    use_gpu = False
    if (torch.cuda.is_available()):
        use_gpu = True
        model.cuda()

    # Loss function and optimizers
    criterion = torch.nn.MSELoss()  # Define MSE loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)  # Use Adam optimizer, use learning_rate hyper parameter

# Initialize low resolution photos
def init_low_res(dir_high_res):
    dir_low_res = dir_high_res + '_low_res'
    img_list = os.listdir(dir_high_res)

    if not os.path.exists(dir_low_res):
        os.makedirs(dir_low_res)

    for img in img_list:
        hr = cv2.imread(dir_high_res + '/' + img, cv2.IMREAD_UNCHANGED)
        hr = np.flip(hr, 2)  # cv2 uses bgr, convert back to rgb
        lr = cv2.resize(hr, (256, 256))
        cv2.imwrite(dir_high_res + '_low_res/' + img, lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # pylint: disable=line-too-long
    # ----------------------------------------------------------------------------
    req_grp = parser.add_argument_group('required')
    req_grp.add_argument('--data_dir', default=None,
                         help='input directory containing the training .tfrecords or images.')
    req_grp.add_argument('--eval_dir', default=None, type=str, help='directory with the validation data.')
    req_grp.add_argument('--log_dir', default=None, help='directory to save checkpoints to.')

    # Training, logging and checkpointing parameters
    train_grp = parser.add_argument_group('training')
    train_grp.add_argument('--learning_rate', default=1e-4, type=float,
                           help='learning rate for the stochastic gradient update.')
    train_grp.add_argument('--weight_decay', default=None, type=float, help='l2 weight decay on FC and Conv layers.')
    train_grp.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
    train_grp.add_argument('--summary_interval', type=int, default=120,
                           help='interval between tensorboard summaries (in s)')
    train_grp.add_argument('--checkpoint_interval', type=int, default=600,
                           help='interval between model checkpoints (in s)')
    train_grp.add_argument('--eval_interval', type=int, default=3600, help='interval between evaluations (in s)')

    # Debug and perf profiling
    # debug_grp = parser.add_argument_group('debug and profiling')
    # debug_grp.add_argument('--profiling', dest='profiling', action='store_true', help='outputs a profiling trace.')
    # debug_grp.add_argument('--noprofiling', dest='profiling', action='store_false')

    # Data pipeline and data augmentation
    data_grp = parser.add_argument_group('data pipeline')
    data_grp.add_argument('--batch_size', default=16, type=int, help='size of a batch for each gradient update.')
    data_grp.add_argument('--data_threads', default=4, help='number of threads to load and enqueue samples.')
    data_grp.add_argument('--rotate', dest="rotate", action="store_true", help='rotate data augmentation.')
    data_grp.add_argument('--flipud', dest="flipud", action="store_true", help='flip up/down data augmentation.')
    data_grp.add_argument('--fliplr', dest="fliplr", action="store_true", help='flip left/right data augmentation.')
    data_grp.add_argument('--random_crop', dest="random_crop", action="store_true",
                          help='random crop data augmentation.')

    # Model parameters
    model_grp = parser.add_argument_group('model_params')
    model_grp.add_argument('--model_name', default=models.__all__[0], type=str, help='classname of the model to use.',
                           choices=models.__all__)
    model_grp.add_argument('--data_pipeline', default='ImageFilesDataPipeline',
                           help='classname of the data pipeline to use.', choices=dp.__all__)
    model_grp.add_argument('--net_input_size', default=256, type=int, help="size of the network's lowres image input.")
    model_grp.add_argument('--output_resolution', default=[512, 512], type=int, nargs=2,
                           help='resolution of the output image.')
    model_grp.add_argument('--batch_norm', dest='batch_norm', action='store_true',
                           help='normalize batches. If False, uses the moving averages.')
    model_grp.add_argument('--nobatch_norm', dest='batch_norm', action='store_false')
    model_grp.add_argument('--channel_multiplier', default=1, type=int,
                           help='Factor to control net throughput (number of intermediate channels).')
    model_grp.add_argument('--guide_complexity', default=16, type=int, help='Control complexity of the guide network.')

    # Bilateral grid parameters
    model_grp.add_argument('--luma_bins', default=8, type=int, help='Number of BGU bins for the luminance.')
    model_grp.add_argument('--spatial_bin', default=16, type=int, help='Size of the spatial BGU bins (pixels).')

    parser.set_defaults(
        profiling=False,
        flipud=False,
        fliplr=False,
        rotate=False,
        random_crop=True,
        batch_norm=False)
    # ----------------------------------------------------------------------------
    # pylint: enable=line-too-long

    args = parser.parse_args()

    model_params = {}
    for a in model_grp._group_actions:
        model_params[a.dest] = getattr(args, a.dest, None)

    data_params = {}
    for a in data_grp._group_actions:
        data_params[a.dest] = getattr(args, a.dest, None)

    run(args, model_params, data_params)
"""
