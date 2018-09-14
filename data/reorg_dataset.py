"""
script to reorganize the dataset to be used by the data loader
run: python /path/to/dataset 
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import os


def organize_data(args):
    # make dirs train, test
    train_dir_input = args.data_dir + '/' + 'train' + '/' + 'input'
    train_dir_output = args.data_dir + '/' + 'train' + '/' + 'output'
    test_dir_input = args.data_dir + '/' + 'test' + '/' + 'input'
    test_dir_output = args.data_dir + '/' + 'test' + '/' + 'output'

    for name in [train_dir_input, train_dir_output, test_dir_input, test_dir_output]:
        os.makedirs(name, exist_ok=True)

    raw = args.data_dir + '/' + 'raw_photos'
    expert_c = args.data_dir + '/' + f'expert_{args.expert}'
    files = os.listdir(raw)

    test = []
    train = []

    for f in files:
        f = Path(f)
        if np.random.rand() < 0.1:
            test.append(f)
            shutil.move(raw + '/' + str(f), test_dir_input + '/' + str(f))
            shutil.move(expert_c + '/' + str(f.stem) + ".jpg", test_dir_output + '/' + str(f.stem) + ".jpg")

        else:
            train.append(f)
            shutil.move(raw + '/' + str(f), train_dir_input + '/' + str(f))
            shutil.move(expert_c + '/' + str(f.stem) + ".jpg", train_dir_output + '/' + str(f.stem) + ".jpg")

    with open(test_dir_input + '/../filelist.txt', 'w+') as fid:
        print(fid)
        for f in map(lambda x: x.split('.')[0], os.listdir(test_dir_input)):
            print(f, file=fid)

    with open(train_dir_input + '/../filelist.txt', 'w+') as fid:
        for f in map(lambda x: x.split('.')[0], os.listdir(train_dir_input)):
            print(f, file=fid)


def organize_data_new(args):
    # make filelists dir
    org_data_dir = 'filelists'
    train_dir = args.data_dir + '/' + org_data_dir + '/' + 'train'
    test_dir = args.data_dir + '/' + org_data_dir + '/' + 'test'

    for name in [train_dir, test_dir]:
        os.makedirs(name, exist_ok=True)

    raw = os.path.abspath(args.data_dir + '/' + 'raw_photos')
    expert = os.path.abspath(args.data_dir + '/' + f'expert_{args.expert}')
    files = os.listdir(raw)

    test_input = []
    test_output = []
    train_input = []
    train_output = []

    for f in files:
        if np.random.rand() < 0.2:
            test_input.append(os.path.join(raw, f))
            test_output.append(os.path.join(expert, f))
        else:
            train_input.append(os.path.join(raw, f))
            train_output.append(os.path.join(expert, f))

    with open(os.path.join(train_dir,'filelist_input.txt'), 'w+') as fid:
        print(fid)
        for f in train_input:
            print(f, file=fid)
    with open(os.path.join(train_dir,'filelist_output.txt'), 'w+') as fid:
        print(fid)
        for f in train_output:
            print(f, file=fid)
    with open(os.path.join(test_dir,'filelist_input.txt'), 'w+') as fid:
        print(fid)
        for f in test_input:
            print(f, file=fid)
    with open(os.path.join(test_dir,'filelist_output.txt'), 'w+') as fid:
        print(fid)
        for f in test_output:
            print(f, file=fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default=None, help='input directory containing the training .tfrecords or images.')
    parser.add_argument('--expert', default='c', choices=['a', 'b', 'c', 'd', 'e'],
                        help='input directory containing the training .tfrecords or images.')
    args = parser.parse_args()

    organize_data_new(args)
