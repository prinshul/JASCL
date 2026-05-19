import torch

seed_current = 1024

BDD_datadir = 'natural_seg/final_dataset/step1/'
IDD_datadir = 'natural_seg/final_dataset/step2/'
step3_datadir = 'natural_seg/final_dataset/step3/'

shots_datadir = 'natural_seg/final_dataset/shots/'



def median_frequency_balance(dataset, num_classes, ignore_index=255, _eps=1e-5):
    '''
    For more details refer to Section 6.3.2 in
    https://arxiv.org/pdf/1411.4734.pdf
    '''
    frequency = torch.zeros(num_classes) + _eps
    for _, seg in dataset:
        for cid in torch.unique(seg):
            if cid == ignore_index:
                continue
            frequency[cid] += torch.sum(seg == cid)
    frequency /= torch.sum(frequency)
    return torch.median(frequency) / frequency


