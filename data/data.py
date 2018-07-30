import logging
import random

import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger("root")
log.setLevel(logging.INFO)


class ImageDataset(Dataset):
    def __init__(self, dir_high_res, dir_edited, list_high_res, list_edited, low_res_init):
        assert len(list_high_res) == len(list_edited), "Uneven number of photos and edited photos"

        # Initialization
        self.dir_high_res = dir_high_res
        self.dir_edited = dir_edited
        self.dir_low_res = dir_high_res + '_low_res'
        self.list_high_res = list_high_res
        self.list_edited = list_edited

    def __len__(self):
        return len(self.list_high_res)

    def __getitem__(self, index):
        # Select sample
        id_high_res = self.list_high_res[index]
        id_edited = self.list_edited[index]

        # Load high-res photo and edited photo
        img_high_res = torch.load(self.dir_high_res + '/' + id_high_res)
        img_low_res = torch.load(self.dir_low_res + '/' + id_high_res)
        img_edited = torch.load(self.dir_edited + '/' + id_edited)

        return img_high_res, img_low_res, img_edited


def check_dir(dirname):
    fnames = os.listdir(dirname)
    if not os.path.isdir(dirname):
        log.error("Training dir {} does not exist".format(dirname))
        return False
    if not "filelist.txt" in fnames:
        log.error("Training dir {} does not containt 'filelist.txt'".format(dirname))
        return False
    if not "input" in fnames:
        log.error("Training dir {} does not containt 'input' folder".format(dirname))
    if not "output" in fnames:
        log.error("Training dir {} does not containt 'output' folder".format(dirname))

    return True


class FivekDataset(Dataset):
    """Pipeline to process pairs of images from a list of image files.
    Assumes path contains:
      - a file named 'filelist.txt' listing the image names.
      - a subfolder 'input'
      - a subfolder 'output'
    """

    def __init__(self, path,
                 output_resolution=(1080, 1920),
                 fliplr=False,
                 flipud=False,
                 rotate=False,
                 random_crop=False,
                 train=False):

        dirname = os.path.dirname(path)
        if not check_dir(dirname):
            raise ValueError("Invalid data path.")
        self.path = path

        with open(self.path, 'r') as fid:
            flist = [l.strip() for l in fid.readlines()]
        self.input_files = [os.path.join(dirname, 'input', f + ".tif") for f in flist]
        self.output_files = [os.path.join(dirname, 'output', f + ".jpg") for f in flist]

        self.train = train
        self.output_resolution = output_resolution

        # Data augmentation
        self.augmentations = [fliplr, flipud, rotate, random_crop]
        self.transform = self.create_transform()
        self.transform_high_res = transforms.Compile([
            transforms.Resize(self.output_resolution, interpolation=2),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, item):
        img_high_res = Image.open(self.input_files[item])
        edited_high_res = Image.open(self.output_files[item])

        # need to set seed to get the same transform for both images
        seed = random.randint(0, 2 ** 32)
        np.random.seed(seed)
        random.seed(seed)
        img_low_res = self.transform(img_high_res)

        np.random.seed(seed)
        random.seed(seed)
        edited_low_res = self.transform(edited_high_res)
        img_high_res = self.transform_high_res(img_high_res)
        edited_high_res = self.transform_high_res(edited_high_res)
        return img_high_res, img_low_res, edited_high_res, edited_low_res

    def create_transform(self, nchan=6):
        """Flip, crop and rotate samples randomly."""

        if self.train:
            augmentation_funcs = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation((0, 90)),
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333),
                                             interpolation=2)
            ]

            trans = [transform for flag, transform in zip(self.augmentations, augmentation_funcs) if flag]
        else:
            trans = []

        trans.extend([
            transforms.Resize(self.output_resolution, interpolation=2),
            transforms.ToTensor(),
            # seems like they aren't normalizing...
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(trans)


def create_loaders(args):
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        FivekDataset(args.train_data,
                     output_resolution=args.imageSize,
                     fliplr=args.fliplr,
                     flipud=args.flipud,
                     rotate=args.rotate,
                     random_crop=args.random_crop,
                     train=True),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        FivekDataset(args.test_data, output_resolution=args.imageSize, train=False),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    return train_loader, test_loader
