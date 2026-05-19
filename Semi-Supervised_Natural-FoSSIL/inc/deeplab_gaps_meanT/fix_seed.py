import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch 

import sys
from shared_quant import seed_current
import random
import numpy as np

random.seed(seed_current)
np.random.seed(seed_current)
torch.manual_seed(seed_current)
torch.cuda.manual_seed(seed_current)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ---- MODEL WRAPPER ----
class MeanTeacherModel(nn.Module):
    def __init__(self, student, teacher):
        super(MeanTeacherModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        print("Mean Teacher Model")
        
    def forward(self, x):
        return self.student(x)

    def update_teacher(self, alpha=0.99):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data




def norm_mean(x):
    # x should be N x F, return 1 x F
    return F.normalize(x, dim=1).mean(dim=0, keepdim=True)





def filter_pseudo_labels(logits, features, proto_dict, confidence_thresh=0.7, similarity_thresh=0.7):
    """
    Perform prototype-based confidence filtering at lower resolution to save GPU memory.

    Args:
        logits: (B, C, H, W) — model output logits
        features: (B, D, H_feat, W_feat) — deep features
        prototypes: (C, D, 1, 1) — class prototypes
        confidence_thresh: float — threshold for prediction confidence
        similarity_thresh: float — threshold for prototype-feature similarity

    Returns:
        filtered_pseudo_labels: (B, H, W) — pseudo labels with -1 for ignored regions
    """
    with torch.no_grad():
        B, C, H, W = logits.shape
        _, D_feat, H_feat, W_feat = features.shape

        # Step 1: Downsample logits to feature map size
        logits_lowres = F.interpolate(logits, size=(H_feat, W_feat), mode='bilinear', align_corners=False)  # (B, C, H_feat, W_feat)

        # Step 2: Compute softmax + pseudo labels
        probs = F.softmax(logits_lowres, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)  # (B, H_feat, W_feat)

        # Step 3: Flatten tensors
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, D_feat)           # (N, D)
        labels_flat = pseudo_labels.view(-1)                                       # (N,)
        max_probs_flat = max_probs.view(-1)                                        # (N,)

        # Step 4: Normalize features and prototypes
        features_flat = F.normalize(features_flat, dim=-1)
        
        prototypes = []
        for label, feat in proto_dict.items():
            prototypes.append(feat)
        
        prototypes = torch.cat(prototypes, dim=0)
        prototypes = prototypes.squeeze(-1).squeeze(-1)

        # Step 5: Class-wise filtering
        similarity = torch.empty_like(max_probs_flat)
        for cls in range(C):
            cls_mask = labels_flat == cls
            if cls_mask.sum() == 0:
                continue
            feats_cls = features_flat[cls_mask]
            proto_cls = prototypes[cls].unsqueeze(0)
            sim_cls = F.cosine_similarity(feats_cls, proto_cls, dim=-1)
            similarity[cls_mask] = sim_cls

        # Step 6: Mask filtering
        keep_mask = (max_probs_flat > confidence_thresh) & (similarity > similarity_thresh)
        filtered_labels_flat = labels_flat.clone()
        filtered_labels_flat[~keep_mask] = C-1

        # Step 7: Reshape to low-res label map
        filtered_lowres_labels = filtered_labels_flat.view(B, H_feat, W_feat)  # (B, H_feat, W_feat)

        # Step 8: Upsample back to original resolution using nearest neighbor
        filtered_pseudo_labels = F.interpolate(
            filtered_lowres_labels.unsqueeze(1).float(), size=(H, W), mode='nearest'
        ).squeeze(1)  # (B, H, W)
        
        
    return filtered_pseudo_labels.unsqueeze(1)







'''
def filter_pseudo_labels(logits, features, proto_dict, confidence_thresh=0.7, similarity_thresh=0.7, metric='cosine'):
    """
    Filters pseudo-labels based on confidence and prototype alignment.

    Args:
        logits: Tensor (1, 6, 512, 1024)
        features: Tensor (1, 304, 128, 256)
        prototypes: Tensor (6, 304, 1, 1)
        confidence_thresh: minimum softmax confidence
        similarity_thresh: minimum similarity to keep a label
        metric: 'cosine' or 'euclidean'

    Returns:
        filtered_pseudo_labels: Tensor (1, 512, 1024), -1 where masked
    """
    B, C, H, W = logits.shape
    _, D_feat, H_feat, W_feat = features.shape

    # Upsample features to match logits resolution
    upsampled_features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)  # shape: (1, 304, 512, 1024)

    # Softmax to get class probabilities
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)  # shape: (1, 512, 1024)

    # Reshape for flattening
    features_flat = upsampled_features.permute(0, 2, 3, 1).reshape(-1, D_feat)  # (512*1024, 304)
    del upsampled_features, logits, features
    labels_flat = pseudo_labels.view(-1)  # (512*1024,)
    max_probs_flat = max_probs.view(-1)  # (512*1024,)
    

    # Get corresponding prototype for each pixel
    # prototypes: (6, 304, 1, 1) → (6, 304)
    prototypes = []
    for label, feat in proto_dict.items():
        prototypes.append(feat)
    prototypes = torch.cat(prototypes, dim=0)
    prototypes = prototypes.squeeze(-1).squeeze(-1)  # (6, 304)
    #proto_flat = prototypes[labels_flat]  # (512*1024, 304)
    
    
    
    if metric == 'cosine':
        similarity = similarity = compute_similarity_chunked(features_flat, labels_flat, prototypes, chunk_size=80000)
        #similarity = F.cosine_similarity(features_flat, proto_flat, dim=-1)  # (512*1024,)
        mask = (max_probs_flat > confidence_thresh) & (similarity > similarity_thresh)
    elif metric == 'euclidean':
        distance = torch.norm(features_flat - proto_flat, dim=-1)
        mask = (max_probs_flat > confidence_thresh) & (distance < similarity_thresh)
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'")

    # Filter invalid labels
    filtered_labels_flat = labels_flat.clone()
    filtered_labels_flat[~mask] = -1

    # Reshape back
    filtered_pseudo_labels = filtered_labels_flat.view(B, H, W)  # (1, 512, 1024)

    return filtered_pseudo_labels
'''




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
    

