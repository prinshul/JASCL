import torch
import numpy as np
import bisect
import os
import os.path as osp
from PIL import Image
import copy

def cumsum(sequence):
    r, s = [], 0
    for e in sequence:
        l = len(e)
        r.append(l + s)
        s += l
    return r


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return idxs


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, labels, indices, mask_label, transform_error=None, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.transform_error = transform_error
        self.target_transform = target_transform
        self.labels = labels
        self.mask_label = mask_label

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise StopIteration
        sample_ = self.dataset[self.indices[idx]]
        if not (sample_['case_name'] in ['1209', '1135', '894', '777']):
            if self.transform is not None:
                # sample = self.transform(sample)
                try:
                    sample = self.transform(sample_.copy())
                except:
                    print(".............Error Raised....................")
                    print("Patient ID: ", sample_['case_name'])

            #     return
                # print(".............Error Raised....................")
                # print("Patient ID: ", sample_['case_name'])
                
                # sample = self.transform_error(sample_.copy())
                # sample['image'] = sample['image'].float()
                # sample['label'] = sample['label'].float()

            if self.mask_label == 'train' and isinstance(sample, list):
                target = self.mask_seg(sample[0]['label'])
                sample[0]['label'] = target
                return sample[0]
            else:
                # print(sample['image'].shape)
                return sample
        else:
            # self.dataset[self.indices[idx]].delete()
            return self.__getitem__(idx + 1)
    def __len__(self):
        return len(self.indices)

    def mask_seg(self, labels, masking_value = 0):
        target = copy.deepcopy(labels)
        lbl_full = np.unique(target)
        for lb in lbl_full:
            if lb not in self.labels:
                target[target == lb] = masking_value
        return target
class ConcatDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MyImageFolder(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        super(MyImageFolder, self).__init__()
        directory = os.path.expanduser(root)
        assert osp.isdir(directory), f"Root must be a directory - {directory}"
        self.files = []
        for name in os.listdir(directory):
            if self.has_file_allowed_extension(name):
                self.files.append(osp.join(directory, name))
        self.transform = transform

    def __getitem__(self, index):
        sample = Image.open(self.files[index]).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 0

    def __len__(self):
        return len(self.files)

    @staticmethod
    def has_file_allowed_extension(filename):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return filename.lower().endswith(extensions)


class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, crop_size=512):
        super().__init__()
        self.size = crop_size

    def __getitem__(self, index):
        sample = torch.randn(3, self.size, self.size)
        return sample, 0

    def __len__(self):
        return 10000
