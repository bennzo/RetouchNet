import torch
from torch.utils.data import Dataset


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
