"""Zappos Dataset utilities."""
import os
from PIL import Image

import torch
from torchvision import transforms


def default_image_loader(path):
    return Image.open(path).convert('RGB')


def check_paths(root, paths):
    """Check whether the passed paths exist or not."""
    existing_files, non_existing_files = list(), list()
    for path in paths:
        if os.path.exists(os.path.join(root, path)):
            existing_files.append(path)
        else:
            non_existing_files.append(path)
    return existing_files, non_existing_files


class ZapposDataset(torch.utils.data.Dataset):

    """ZapposDataset return image and condition."""

    def __init__(self, root, base_path, files_json_path, condition,
                 conditions=None, transform=None, loader=default_image_loader):
        self.root = root
        self.base_path = base_path
        self.img_root = os.path.join(self.root, self.base_path)
        with open(os.path.join(self.root, files_json_path)) as f:
            paths = [line.rstrip('\n') for line in f]

        self.files, self.non_existing_files = check_paths(self.img_root, paths)
        if len(self.non_existing_files) > 0:
            print('\t+{} non existing files\n'.format(len(self.non_existing_files)))
        self.condition = condition
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.img_root, self.files[index])
        if os.path.exists(path):
            img = self.loader(path)
            if self.transform is not None:
                return self.transform(img), self.condition
        else:
            return None

    def split(self, split):
        return self.all_files[split]


def make_dataset(condition, root, base_path, files_json_path):
    """Make a zappos dataset class

    Args:
        condition: for Zappos, should be in [0, 3].
        root: root directory having zappos directory in it.
        base_path: directory name of zappos dataset.
        files_json_path: list of relative paths to images from base_path.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ZapposDataset(root, base_path, files_json_path, condition,
                            transform=transforms.Compose([
                                transforms.Scale(112),
                                transforms.CenterCrop(112),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    return dataset


def make_data_loader(condition, root='data', base_path='ut-zap50k-images',
                     files_json_path='filenames.json',
                     batch_size=64, shuffle=False, **kwargs):
    """Make a loader of ZapposDataset.

    Args:
        root: path to directory including files_json.
        base_path: directory name of ut-zap50k-images.
        files_json_path: file contains all images' relative path from
            base_path.
        batch_size: default size is 64. crcrpar confirmed that 256 is too
            large for 8GB memory.
        shuffle: default is False because I assume this loader is used
            for feature extranciton, not training.
    """
    dataset = make_dataset(condition, root, base_path, files_json_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, **kwargs)
    return loader
