import argparse
import cv2
import os
import numpy as np


def run(args):
    if (args['low_res_init']):
        init_low_res(args['dir_high_res'])

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

    ## Define arguments
    # Mandatory
    mand_group = parser.add_argument_group('mandatory')
    mand_group.add_argument('dir_high_res', default=None, help='Directory path of the untouched high-res photos')
    mand_group.add_argument('dir_edited', default=None, help='Directory path of the edited high-res photos')
    mand_group.add_argument('low_res_init', default=False, help='Initialization of the low-res photos')

    # Optional
    ## TODO: Add optional arguments such as dataloader, optimizer, network args

    ## Parse arguments
    args = parser.parse_args()

    mand_params = {}
    for arg in mand_group._group_actions:
        mand_params[arg.dest] = getattr(args, arg.dest, None)

    ## Run
    run(mand_params)

