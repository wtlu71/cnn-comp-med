import h5py
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np

class H5Dataset(Dataset):
    """
    custom dataloading class that inherits from torch.utils.data Dataset to load .h5 files from PatchCamelyon project
    """
    def __init__(self, img_h5_path,label_h5_path, transform=None, img_key="x", label_key="y"):
        self.img_h5_path = img_h5_path 
        self.label_h5_path = label_h5_path
        self.transform = transform
        self.img_key = img_key # in these files 
        self.label_key = label_key
        self.img_h5_file = None  # lazy open
        self.label_h5_file = None  # lazy open

    def _ensure_open(self):
        if self.img_h5_file is None or self.label_h5_file is None:
            self.img_h5_file = h5py.File(self.img_h5_path, "r")
            self.label_h5_file = h5py.File(self.label_h5_path, "r")
            self.imgs = self.img_h5_file[self.img_key]
            self.labels = self.label_h5_file[self.label_key]

    def __len__(self):
        with h5py.File(self.img_h5_path, "r") as f:
            return len(f[self.img_key])
        # self._ensure_open()
        # return len(self.imgs)

    # inherits __getitem__ from torch Dataset and overwrites it for our purposes
    def __getitem__(self, idx):
        self._ensure_open()
        img = self.imgs[idx]
        # unnecessary b/c of 
        img = np.transpose(img, (2,0,1))
        # normalize from RGBs from 0 to 255 to float between 0 and 1
        img = torch.from_numpy(img.astype("float32") / 255.0)  # HWC -> CHW
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img,label
