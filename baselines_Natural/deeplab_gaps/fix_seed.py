import numpy as np
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


def overlay_mask(img1, lbl1, img2, lbl2, labels, alpha=1):
    """
    Overlay a multi-label segmentation masks corresponding image on an RGB image.

    Parameters:
        image (numpy.ndarray): The RGB image.
        mask (numpy.ndarray): The multi-label segmentation mask.

    Returns:
        numpy.ndarray: The image with the overlay.
    """
    over_img = img1.copy()
    over_lbl = lbl1.copy()
    print("Old labels = ",np.unique(lbl1), np.unique(lbl2))

    for label in labels:
        if label == 255:  # Skip background
            continue
        mask_label = (lbl2 == label)
        #print(label,np.unique(mask_label))
        over_img[mask_label]=img2[mask_label]
        over_lbl[mask_label] = label

    return over_img, over_lbl
    

