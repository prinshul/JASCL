import torch
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from utils.loss import KnowledgeDistillationLoss, CosineLoss, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, CosineKnowledgeDistillationLoss
from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier, CosineClassifier, SPNetClassifier
from .utils import get_scheduler, MeanReduction, get_prototype
from networks.VerSe.unet import network as network
from torch.nn.modules.loss import CrossEntropyLoss
from utils.VerSe_utils import DiceLoss, print_network
from .svf import resolver
import SimpleITK as sitk
import numpy as np
import os
import random
import pickle


CLIP = 10

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def calculate_certainty(sigmoid_outputs):
    """
    Calculate prediction certainty/uncertainty map
    Args:
        sigmoid_outputs: [N, C, D, H, W] - sigmoid probabilities
    Returns:
        certainty: [N, 1, D, H, W] - certainty map (higher = more certain)
    """
    # Method 1: Maximum probability
    max_probs, _ = sigmoid_outputs.max(dim=1, keepdim=True)  # [N, 1, D, H, W]
    
    # Method 2: Entropy-based (lower entropy = higher certainty)
    # Normalize to get proper probability distribution
    eps = 1e-10
    probs = sigmoid_outputs / (sigmoid_outputs.sum(dim=1, keepdim=True) + eps)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1, keepdim=True)
    max_entropy = np.log(sigmoid_outputs.shape[1])  # Maximum possible entropy
    certainty_entropy = 1 - (entropy / max_entropy)  # Normalize to [0, 1]
    
    # Combine both methods
    certainty = (max_probs + certainty_entropy) / 2.0
    
    return certainty


class PKDLoss(nn.Module):
    """Pixel-wise Knowledge Distillation Loss - adapted for 3D"""
    def __init__(self):
        super().__init__()
        self.call_count = 0
    
    def forward(self, features_new, features_old, pseudo_label_region):
        """
        features_new: list of feature maps from new model
        features_old: list of feature maps from old model  
        pseudo_label_region: mask indicating regions with pseudo labels [N, 1, D, H, W]
        """
        self.call_count += 1
        
        loss = 0
        for idx, (feat_new, feat_old) in enumerate(zip(features_new, features_old)):
            # Normalize features
            feat_new_norm = F.normalize(feat_new, p=2, dim=1)
            feat_old_norm = F.normalize(feat_old, p=2, dim=1)
            
            # Resize pseudo_label_region to match feature dimensions
            if pseudo_label_region.shape[-3:] != feat_new.shape[-3:]:
                pseudo_label_region_resized = F.interpolate(
                    pseudo_label_region, 
                    size=feat_new.shape[-3:], 
                    mode='nearest'
                )
            else:
                pseudo_label_region_resized = pseudo_label_region
            
            # Compute cosine similarity
            similarity = (feat_new_norm * feat_old_norm).sum(dim=1, keepdim=True)
            
            # Only compute loss on pseudo-labeled regions
            masked_loss = ((1 - similarity) * pseudo_label_region_resized).sum() / \
                         (pseudo_label_region_resized.sum() + 1e-6)
            loss += masked_loss
            
            if self.call_count % 50 == 0:
                print(f"    [PKD] Layer {idx}: similarity_mean={similarity.mean().item():.4f}, "
                      f"masked_loss={masked_loss.item():.4f}")
        
        final_loss = loss / len(features_new)
        
        if self.call_count % 50 == 0:
            print(f"    [PKD] Total loss: {final_loss.item():.4f} (averaged over {len(features_new)} layers)")
        
        return final_loss


class ContrastiveLoss(nn.Module):
    """Contrastive Prototype Discriminative (CPD) Loss - separate new classes from old prototypes"""
    def __init__(self, n_old_classes, n_new_classes):
        super().__init__()
        self.n_old_classes = n_old_classes
        self.n_new_classes = n_new_classes
        self.call_count = 0
    
    def forward(self, features, logits_new, labels, n_new_classes, old_prototypes):
        """
        features: [N, C, D, H, W] - feature map from encoder
        logits_new: [N, total_classes, D, H, W] - all logits
        labels: [N, D, H, W] - ground truth labels
        n_new_classes: number of new classes in current step
        old_prototypes: [n_old, C] - prototypes of old classes
        """
        self.call_count += 1
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)  # [N, C, D, H, W]
        old_prototypes_norm = F.normalize(old_prototypes, p=2, dim=1)  # [n_old, C]
        
        loss = 0
        count = 0
        
        # For each new class in the current labels
        unique_labels = labels.unique()
        
        if self.call_count % 50 == 0:
            print(f"    [CPD] Processing {len(unique_labels)} unique labels: {unique_labels.cpu().numpy()}")
        
        # Determine which labels are "new" classes
        # Assuming labels are continuous and new classes are at the end
        n_total_old = old_prototypes.shape[0]
        
        for cls_idx in unique_labels:
            if cls_idx == 0 or cls_idx == 255:  # Skip background and ignore index
                continue
            
            # Check if this is a new class (beyond old classes)
            if cls_idx <= n_total_old:
                continue  # Skip old classes
            
            # Get mask for current class
            mask = (labels == cls_idx)  # [N, D, H, W]
            if mask.sum() == 0:
                continue
            
            # Get features for this class
            mask_expanded = mask.unsqueeze(1).float()  # [N, 1, D, H, W]
            
            # Extract features for this class and flatten
            class_features = features_norm * mask_expanded  # [N, C, D, H, W]
            class_features = class_features.permute(1, 0, 2, 3, 4)  # [C, N, D, H, W]
            class_features = class_features.reshape(class_features.shape[0], -1)  # [C, N*D*H*W]
            class_features = class_features[:, mask.flatten()]  # [C, num_pixels]
            
            # Compute similarity with all old prototypes
            similarities = torch.mm(old_prototypes_norm, class_features)  # [n_old, num_pixels]
            
            # We want to minimize similarity (maximize dissimilarity)
            # Use margin-based loss: push similarity below threshold
            margin = 0.5
            class_loss = F.relu(similarities - margin).mean()
            loss += class_loss
            count += 1
            
            if self.call_count % 50 == 0:
                print(f"      [CPD] Class {cls_idx.item()}: {mask.sum().item()} pixels, "
                      f"max_similarity={similarities.max().item():.4f}, loss={class_loss.item():.4f}")
        
        final_loss = loss / max(count, 1)
        
        if self.call_count % 50 == 0:
            print(f"    [CPD] Total loss: {final_loss.item():.4f} (averaged over {count} classes)")
        
        return final_loss


class WBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss"""
    def __init__(self, pos_weight=None, n_old_classes=0, n_new_classes=1):
        super().__init__()
        self.pos_weight = pos_weight
        self.n_old_classes = n_old_classes
        self.n_new_classes = n_new_classes
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [N, C, D, H, W] - raw logits for new classes
            labels: [N, D, H, W] - ground truth labels or [N, N_samples, 1] for fake features
        Returns:
            loss: [C] - loss per class
        """
        if labels.dim() == 3:  # Fake features case: [N, N_samples, 1]
            # Create one-hot encoding
            N, C, N_samples = logits.shape[:3]
            target = torch.zeros_like(logits).float()  # [N, C, N_samples, ...]
            # All fake features should be classified as background (class 0)
            # So all new class logits should be low (target = 0)
        else:  # Real images case: [N, D, H, W]
            # Convert labels to one-hot for new classes
            N, C, D, H, W = logits.shape
            target = torch.zeros_like(logits).float()
            
            for c_idx in range(C):
                class_label = self.n_old_classes + c_idx + 1
                target[:, c_idx] = (labels == class_label).float()
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)  # Clamp to safe range
        
        # Binary cross-entropy
        if self.pos_weight is not None:
            loss = -(self.pos_weight * target * torch.log(probs) + 
                    (1 - target) * torch.log(1 - probs))
        else:
            loss = -(target * torch.log(probs) + 
                    (1 - target) * torch.log(1 - probs))
        
        return loss


class Trainer:
    def __init__(self, task, device, logger, opts, config_vit):
        self.logger = logger
        self.device = device
        self.c1 = 0.7
        self.c2 = 0.3
        self.task = task
        self.opts = opts
        self.novel_classes = self.task.get_n_classes()[-1]
        self.labels_old = task.get_old_labels(bkg=False)
       
        self.step = task.step        
        self.labels = self.task.get_novel_labels()
        self.total_classes = self.labels + self.labels_old
        self.need_model_old = (self.opts.born_again or self.opts.mib_kd > 0 or self.opts.loss_kd > 0 or
                               self.opts.born_again or self.opts.l2_loss > 0 or self.opts.l1_loss > 0 or 
                               self.opts.cos_loss > 0 or self.step > 0)

        self.n_channels = -1
        self.model = self.make_model(config_vit)
        self.model = self.model.to(device)
        self.svf = opts.svf
        self.distributed = False
        self.model_old = None

        if self.opts.fix_bn:
            self.model.fix_bn()

        if self.opts.bn_momentum is not None:
            self.model.bn_set_momentum(self.opts.bn_momentum)

        self.initialize(self.opts)

        self.born_again = self.opts.born_again
        self.dist_warm_start = self.opts.dist_warm_start
        model_old_as_new = self.opts.born_again or self.opts.dist_warm_start
        
        if self.need_model_old:
            self.model_old = self.make_model(config_vit, is_old=not model_old_as_new)
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.to(device)
            self.model_old.eval()

        self.train_only_novel = self.opts.train_only_novel

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8, 
                                          betas=(0.9, 0.999), weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 25, eta_min=5e-6)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)
        
        # ============= Adaptive Prototype Replay Components =============
        if self.step > 0 and self.model_old is not None:
            # Loss functions
            self.pkd_criterion = PKDLoss()
            self.cont_criterion = ContrastiveLoss(
                n_old_classes=len(self.labels_old),
                n_new_classes=len(self.labels)
            )
            self.uncertainty_criterion = nn.MSELoss(reduction='mean')
            
            # Loss weights
            self.pkd_weight = getattr(opts, 'pkd_weight', 1.0)
            self.cont_weight = getattr(opts, 'cont_weight', 0.1)
            self.uncertainty_weight = getattr(opts, 'uncertainty_weight', 0.5)
            
            # BCE losses for different components
            self.bce_criterion = WBCELoss(
                n_old_classes=len(self.labels_old),
                n_new_classes=len(self.labels)
            )
            self.bce_fake_criterion = WBCELoss(
                n_old_classes=len(self.labels_old),
                n_new_classes=len(self.labels)
            )
            self.bce_extra_bg_criterion = WBCELoss(
                n_old_classes=len(self.labels_old),
                n_new_classes=len(self.labels)
            )
            
            self.logger.info("=" * 80)
            self.logger.info("ADAPTIVE PROTOTYPE REPLAY INITIALIZED")
            self.logger.info(f"  PKD weight: {self.pkd_weight}")
            self.logger.info(f"  Contrastive weight: {self.cont_weight}")
            self.logger.info(f"  Uncertainty weight: {self.uncertainty_weight}")
            self.logger.info("=" * 80)
        else:
            self.pkd_criterion = None
            self.cont_criterion = None
       
        # Feature distillation
        if opts.l2_loss > 0 or opts.cos_loss > 0 or opts.l1_loss > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.l2_loss > 0:
                self.feat_loss = opts.l2_loss
                self.feat_criterion = nn.MSELoss()
            elif opts.l1_loss > 0:
                self.feat_loss = opts.l1_loss
                self.feat_criterion = nn.L1Loss()
            elif opts.cos_loss > 0:
                self.feat_loss = opts.cos_loss
                self.feat_criterion = CosineLoss()
        else:
            self.feat_criterion = None
        
        # Output distillation
        if opts.loss_kd > 0 or opts.mib_kd > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.loss_kd > 0:
                if opts.ckd:
                    self.kd_criterion = CosineKnowledgeDistillationLoss(reduction='mean')
                else:
                    self.kd_criterion = KnowledgeDistillationLoss(reduction="mean", alpha=opts.kd_alpha)
                self.kd_loss = opts.loss_kd
            if opts.mib_kd > 0:
                self.kd_loss = opts.mib_kd
                self.kd_criterion = UnbiasedKnowledgeDistillationLoss(reduction="mean")
        else:
            self.kd_criterion = None
        
        # Body distillation
        if opts.loss_de > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            self.de_loss = opts.loss_de
            self.de_criterion = nn.MSELoss()
        else:
            self.de_criterion = None
        
        # Prototype tracking for ADC
        self.prev_prototypes = None
        self.prev_proto_stats = None
        self.true_prototypes = None  # For refinement tracking
        self.pixel_counts = None  # Track pixel counts per class
            
    def make_model(self, config_vit, is_old=False):
        n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
        model = make_model(self.opts, config_vit, n_classes[0])
        return model

    def distribute(self):
        if self.model is not None:
            self.distributed = True
            self.model = DistributedDataParallel(self.model, device_ids=[self.opts.device_id],
                                                 output_device=self.opts.device_id, 
                                                 find_unused_parameters=True)

    def get_classifier(self, is_old=False):
        self.opts = self.opts
        if self.opts.method == "SPN":
            classes = self.task.get_old_labels() if is_old else self.task.get_order()
            cls = SPNetClassifier(self.opts, classes)
            n_feat = cls.channels
        elif self.opts.method == 'COS':
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = CosineClassifier(n_classes, channels=n_feat)
        else:
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = IncrementalClassifier(n_classes, channels=n_feat)
        return cls, n_feat

    def initialize(self, opts):
        if self.opts.init_mib and self.opts.method == "FT":
            device = self.device
            model = self.model.module if self.distributed else self.model

            classifier = model.cls
            imprinting_w = classifier.cls[0].weight[0]
            bkg_bias = classifier.cls[0].bias[0]

            bias_diff = torch.log(torch.FloatTensor([self.task.get_n_classes()[-1] + 1])).to(device)
            new_bias = (bkg_bias - bias_diff)

            classifier.cls[-1].weight.data.copy_(imprinting_w)
            classifier.cls[-1].bias.data.copy_(new_bias)
            classifier.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def _compute_prototype_statistics(self, protos):
        """Compute statistics for each prototype (mean norm, std, noise)"""
        stats = {}
        
        self.logger.info("=" * 80)
        self.logger.info("COMPUTING PROTOTYPE STATISTICS")
        self.logger.info("=" * 80)
        
        for cls_idx, proto in protos.items():
            if not isinstance(cls_idx, int):
                continue
            
            # proto shape: [C, 1, 1, 1] or [C]
            if proto.dim() == 4:
                proto = proto.squeeze(-1).squeeze(-1).squeeze(-1)
            
            # Compute norm statistics
            norm_mean = torch.norm(proto, p=2).item()
            norm_std = norm_mean * 0.1  # Simplified: assume 10% std
            
            # Compute noise (variance of prototype)
            noise_std = proto.std().item()
            
            stats[cls_idx] = {
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'noise_std': noise_std
            }
            
            self.logger.info(f"  Class {cls_idx}: norm_mean={norm_mean:.4f}, "
                           f"norm_std={norm_std:.4f}, noise_std={noise_std:.4f}")
        
        self.logger.info(f"Total prototypes with statistics: {len(stats)}")
        self.logger.info("=" * 80)
        
        return stats

    def _generate_fake_features(self, protos, proto_stats, samples_per_class):
        """Generate fake features from prototypes with noise (Adaptive Prototype Replay)"""
        fake_features_list = []
        
        self.logger.debug(f"[Fake Features] Generating {samples_per_class} samples per class")
        
        for cls_idx in sorted([k for k in protos.keys() if isinstance(k, int)]):
            proto = protos[cls_idx]
            
            # Get prototype shape [C, 1, 1, 1] or reshape to it
            if proto.dim() == 1:
                proto = proto.view(-1, 1, 1, 1)
            elif proto.dim() == 4:
                pass
            else:
                proto = proto.view(-1, 1, 1, 1)
            
            C = proto.shape[0]
            
            # Generate multiple samples for this class
            # Shape: [C, 1, N_samples, 1]
            per_cls_fake_features = proto.repeat(1, 1, samples_per_class, 1)
            
            # Add noise based on statistics
            if proto_stats is not None and cls_idx in proto_stats:
                noise_std = proto_stats[cls_idx]['noise_std']
                noise = torch.randn_like(per_cls_fake_features) * noise_std
                per_cls_fake_features = per_cls_fake_features + noise
                
                # Scale by norm statistics (random scaling around mean norm)
                norm_mean = proto_stats[cls_idx]['norm_mean']
                norm_std = proto_stats[cls_idx]['norm_std']
                rand_norm = torch.randn(1, 1, samples_per_class, 1).to(proto.device) * norm_std + norm_mean
                per_cls_fake_features = F.normalize(per_cls_fake_features, p=2, dim=0) * rand_norm
                
                self.logger.debug(f"  Class {cls_idx}: generated {samples_per_class} samples")
            
            fake_features_list.append(per_cls_fake_features)
    
        if len(fake_features_list) > 0:
            fake_features = torch.cat(fake_features_list, dim=2)  # [C, 1, total_samples, 1]
            # Remove singleton dimensions and reshape to [1, C, total_samples, 1, 1] for Conv3d
            fake_features = fake_features.squeeze(1).squeeze(-1)  # [C, total_samples]
            fake_features = fake_features.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, C, total_samples, 1, 1]
            fake_features = fake_features.to(self.device)
            
            self.logger.debug(f"[Fake Features] Total shape: {fake_features.shape}")
            
            return fake_features
        
        self.logger.warning("[Fake Features] No fake features generated!")
        return None

    def _compute_pred_numbers(self, train_loader):
        """Compute predicted pixel counts for each old class on new data (for ADC)"""
        self.logger.info("=" * 80)
        self.logger.info("COMPUTING PREDICTED PIXEL NUMBERS FOR ADC")
        self.logger.info("=" * 80)
        
        n_old_classes = len(self.labels_old)
        pred_numbers = torch.zeros(n_old_classes + 1).to(self.device)
        
        self.model_old.eval()
        with torch.no_grad():
            for batch_idx, sampled_batch in enumerate(train_loader):
                image_batch = sampled_batch['image'].to(self.device)
                label_batch = sampled_batch['label'].to(self.device)
                
                # Get old model predictions
                logits_old = self.model_old(image_batch)
                pred_old = logits_old.argmax(dim=1)  # [N, D, H, W]
                
                # Count pixels at downsampled resolution (stride 16)
                pred_region = (pred_old * (label_batch == 0))[:, ::16, ::16, ::16]
                real_bg_region = torch.logical_and(pred_old == 0, label_batch == 0)[:, ::16, ::16, ::16]
                
                pred_numbers[0] += real_bg_region.sum()
                
                for i in range(1, n_old_classes + 1):
                    pred_numbers[i] += (pred_region == i).sum()
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
        
        self.logger.info("Predicted pixel numbers:")
        for i, count in enumerate(pred_numbers):
            self.logger.info(f"  Class {i}: {count.item():.0f} pixels")
        self.logger.info("=" * 80)
        
        return pred_numbers

    def _refine_prototypes_adc(self, train_loader):
        """
        Adaptive Deviation Compensation (ADC): Refine prototypes based on representation shift
        This is the core innovation of the Adapter method
        """
        self.logger.info("=" * 80)
        self.logger.info("ADAPTIVE DEVIATION COMPENSATION (ADC) - REFINING PROTOTYPES")
        self.logger.info("=" * 80)
        
        n_old_classes = len(self.labels_old)
        
        # Initialize accumulators
        pred_numbers = torch.zeros(n_old_classes + 1).to(self.device)
        pred_numbers_new = torch.zeros(n_old_classes + 1).to(self.device)
        prototypes_old = torch.zeros(n_old_classes, 256).to(self.device)  # Adjust feature dim
        prototypes_new = torch.zeros(n_old_classes, 256).to(self.device)
        
        self.model.eval()
        self.model_old.eval()
        
        with torch.no_grad():
            for batch_idx, sampled_batch in enumerate(train_loader):
                image_batch = sampled_batch['image'].to(self.device)
                label_batch = sampled_batch['label'].to(self.device)
                
                # Get features from both models
                features_old = self.model_old(image_batch, ret_intermediate=True)
                features_new = self.model(image_batch, ret_intermediate=True)

                # Handle different return formats
                if isinstance(features_old, tuple):
                    features_old = features_old[1] if len(features_old) > 1 else features_old[0]
                if isinstance(features_new, tuple):
                    features_new = features_new[1] if len(features_new) > 1 else features_new[0]

                # Use last feature map
                feat_old = features_old[-1] if isinstance(features_old, list) else features_old
                feat_new = features_new[-1] if isinstance(features_new, list) else features_new
                
                # Normalize features
                normalized_feat_old = F.normalize(feat_old, p=2, dim=1)
                normalized_feat_new = F.normalize(feat_new, p=2, dim=1)
                
                # Get predictions from old model
                logits_old = self.model_old(image_batch)
                pred_old = logits_old.argmax(dim=1)  # [N, D, H, W]
                
                # Get predictions from new model
                logits_new = self.model(image_batch)
                pred_new = logits_new.argmax(dim=1)
                
                # Calculate certainty/uncertainty
                uncer_old = calculate_certainty(torch.sigmoid(logits_old)).squeeze(1)[:, ::16, ::16, ::16]
                uncer_new = calculate_certainty(torch.sigmoid(logits_new)).squeeze(1)[:, ::16, ::16, ::16]
                
                # Downsample predictions
                pred_region_old = (pred_old * (label_batch == 0))[:, ::16, ::16, ::16]
                pred_region_new = (pred_new * (label_batch == 0))[:, ::16, ::16, ::16]
                
                # Downsample features
                feat_old_down = normalized_feat_old[:, :, ::16, ::16, ::16]
                feat_new_down = normalized_feat_new[:, :, ::16, ::16, ::16]
                
                # Accumulate prototypes for pixels that both models agree on with high certainty
                for cls_idx in range(1, n_old_classes + 1):
                    # Mask: both models predict same class AND both are certain
                    mask = ((pred_region_old == cls_idx) & 
                           (pred_region_new == cls_idx) & 
                           (uncer_old > 0.7) & 
                           (uncer_new > 0.7))
                    
                    if mask.sum() > 0:
                        # Extract features where mask is True
                        mask_expanded = mask.unsqueeze(1)  # [N, 1, D, H, W]
                        
                        # Accumulate old model features
                        class_feat_old = (feat_old_down * mask_expanded.float()).sum(dim=[0, 2, 3, 4])
                        prototypes_old[cls_idx - 1] += class_feat_old
                        
                        # Accumulate new model features
                        class_feat_new = (feat_new_down * mask_expanded.float()).sum(dim=[0, 2, 3, 4])
                        prototypes_new[cls_idx - 1] += class_feat_new
                        
                        # Count pixels
                        pred_numbers[cls_idx] += mask.sum()
                    
                    # Also count all old predictions (for ratio calculation)
                    pred_numbers_new[cls_idx] += (pred_region_old == cls_idx).sum()
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
        
        # Normalize accumulated prototypes
        prototypes_old = F.normalize(prototypes_old, p=2, dim=1)
        prototypes_new = F.normalize(prototypes_new, p=2, dim=1)
        
        # Calculate representation shift (bias)
        prototypes_bias = prototypes_new - prototypes_old
        
        # Calculate update ratio
        ratio = pred_numbers[1:] / (self.pixel_counts[1:n_old_classes+1] + 1e-6)
        ratio = torch.clamp(ratio, 0, 1)
        
        # Update prototypes based on ratio
        self.logger.info("Updating prototypes with ADC:")
        for i in range(n_old_classes):
            if prototypes_old[i].sum() == 0 or prototypes_new[i].sum() == 0:
                self.logger.info(f"  Class {i+1}: No update (insufficient data)")
                continue
            
            old_proto = self.prev_prototypes[i].clone()
            current_proto = self.true_prototypes[i] + prototypes_bias[i]
            updated_proto = (1 - ratio[i]) * self.prev_prototypes[i] + ratio[i] * current_proto
            
            self.prev_prototypes[i] = updated_proto
            
            shift_magnitude = torch.norm(prototypes_bias[i]).item()
            self.logger.info(f"  Class {i+1}: ratio={ratio[i].item():.3f}, shift={shift_magnitude:.4f}")
        
        # Update noise estimates
        for i in range(n_old_classes):
            noise_increase = torch.abs(self.prev_prototypes[i] - self.true_prototypes[i])
            if self.prev_proto_stats and (i+1) in self.prev_proto_stats:
                old_noise = self.prev_proto_stats[i+1]['noise_std']
                new_noise = old_noise + noise_increase.mean().item()
                self.prev_proto_stats[i+1]['noise_std'] = new_noise
                self.logger.info(f"  Class {i+1}: noise updated {old_noise:.4f} -> {new_noise:.4f}")
        
        self.logger.info("=" * 80)

    def warm_up(self, dataset, epochs=1):
        self.warm_up_(dataset, epochs)
        if self.dist_warm_start:
            self.model_old.load_state_dict(self.model.state_dict())

    def warm_up_(self, dataset, epochs=1):
        pass

    def cool_down(self, dataset, epochs=1):
        pass
    
    def visualize_pred(self, labels, prediction, cur_epoch, cur_step):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        
        os.makedirs('data/merged/dataset/prediction/ground_truth', exist_ok=True)
        os.makedirs('data/merged/dataset/prediction/predicted_mask', exist_ok=True)
        
        sitk.WriteImage(label_arr, os.path.join('data/merged/dataset/prediction/ground_truth',
                       f'{cur_epoch}_{cur_step}_{self.opts.network_arch}.nii.gz'))
        sitk.WriteImage(prediction_arr, os.path.join('data/merged/dataset/prediction/predicted_mask',
                       f'{cur_epoch}_{cur_step}_{self.opts.network_arch}.nii.gz'))

    def train(self, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, 
              print_int=10, n_iter=1, snapshot_path=None):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = optimizer
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        
        num_classes = self.opts.num_classes
        max_iterations = self.opts.max_epochs * len(train_loader)
        max_epoch = self.opts.max_epochs
        iter_num = cur_epoch * len(train_loader)
        model.train()
        
        # Freeze BN and dropout for incremental learning
        if self.step > 0:
            if hasattr(model, 'freeze_bn'):
                model.freeze_bn(affine_freeze=False)
            if hasattr(model, 'freeze_dropout'):
                model.freeze_dropout()
        
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.opts.lr, momentum=0.9, 
                                    weight_decay=0.0001, nesterov=True)

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0

        # ============= Load Previous Prototypes and Initialize ADC =============
        prev_protos = None
        prev_proto_stats = None
        
        if self.step > 0:
            logger.info("=" * 80)
            logger.info("LOADING PREVIOUS PROTOTYPES FOR STEP %d" % self.step)
            
            proto_path = '../../../base/codu_run/codu/saved_proto.pkl' if self.step == 1 else 'saved_proto.pkl'
            
            if os.path.exists(proto_path):
                with open(proto_path, 'rb') as f:
                    prev_protos = pickle.load(f)
                logger.info(f"  Loaded prototypes from: {proto_path}")
                logger.info(f"  Number of prototypes: {len([k for k in prev_protos.keys() if isinstance(k, int)])}")
                
                # Compute prototype statistics
                prev_proto_stats = self._compute_prototype_statistics(prev_protos)
                
                # Store for ADC
                self.prev_prototypes = torch.stack([
                    prev_protos[k].squeeze() for k in sorted([k for k in prev_protos.keys() if isinstance(k, int)])
                ]).to(device)
                self.true_prototypes = self.prev_prototypes.clone()
                self.prev_proto_stats = prev_proto_stats
                
                logger.info(f"  Prototypes tensor shape: {self.prev_prototypes.shape}")
            else:
                logger.warning(f"  Prototype file not found: {proto_path}")
                logger.warning("  ADC will be disabled for this training")
            
            logger.info("=" * 80)
        
        # ============= Compute Predicted Numbers (First Epoch Only) =============
        if self.step > 0 and cur_epoch == 0 and prev_protos is not None:
            pred_numbers = self._compute_pred_numbers(train_loader)
            
            # Calculate samples per iteration for fake feature generation
            n_old_classes = len([k for k in prev_protos.keys() if isinstance(k, int)])
            total_old_pixels = pred_numbers[1:].sum()
            
            # Determine how many fake samples to generate per class per iteration
            self.samples_per_class_per_iter = max(1, int(100 / n_old_classes))
            
            # Store pixel counts for ADC
            self.pixel_counts = pred_numbers.clone()
            
            logger.info(f"Fake samples per class per iteration: {self.samples_per_class_per_iter}")
        
        # ============= Training Loop =============
        for i_batch, sampled_batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            if self.step == 0:
                # ============= BASE STEP - Standard Training =============
                feat, logits = model(image_batch)
                
                loss_ce_1 = ce_loss(logits, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    logits, label_batch[:].long(), self.total_classes, softmax=True)

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss_tot = loss
                dice = 1 - loss_dice_1

                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()
            
            else:
                # ============= INCREMENTAL STEP - Adaptive Prototype Replay =============
                
                # Generate fake features from old prototypes
                fake_features = None
                if prev_protos is not None and hasattr(self, 'samples_per_class_per_iter'):
                    fake_features = self._generate_fake_features(
                        prev_protos, prev_proto_stats, self.samples_per_class_per_iter
                    )
                
                # Get old model outputs for pseudo-labeling and PKD
                with torch.no_grad():
                    features_old = self.model_old(image_batch, ret_intermediate=True)
                    logits_old = self.model_old(image_batch)
                    if isinstance(features_old, tuple):
                        features_old = features_old[1] if len(features_old) > 1 else features_old[0]
                    
                    pred_old = logits_old.argmax(dim=1)  # [N, D, H, W]
                    
                    # Create pseudo-label region: background in GT but predicted as class by old model
                    pseudo_label_region = torch.logical_and(
                        label_batch == 0,
                        pred_old > 0
                    ).unsqueeze(1).float()  # [N, 1, D, H, W]
                
                # Detect background regions for extra background sampling
                region_bg = torch.logical_and(pred_old == 0, label_batch == 0)[:, ::16, ::16, ::16]
                
                # Forward pass with fake features and extra BG
                logits, features, extra = model(
                    image_batch, 
                    fake_features=fake_features,
                    region_bg=region_bg,
                    ret_intermediate=True
                )
                logits_fake, logits_extra_bg = extra
                
                # ============= Compute Losses =============
                
                # 1. Standard CE + Dice loss on real data
                loss_ce_1 = ce_loss(logits, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    logits, label_batch[:].long(), self.total_classes, softmax=True)
                
                # 2. Modified BCE Loss (MBCE) on real images
                # 2. Modified BCE Loss (MBCE) on real images
                n_new_classes = len(self.labels)
                n_old_classes = len(self.labels_old)

                logits_new_classes = logits[:, n_old_classes:, :, :, :]

                loss_mbce_ori = self.bce_criterion(logits_new_classes, label_batch).mean(dim=[0, 2, 3, 4])
                if torch.isnan(loss_mbce_ori).any() or torch.isinf(loss_mbce_ori).any():
                    logger.warning(f"NaN in loss_mbce_ori, setting to 0")
                    loss_mbce_ori = torch.zeros_like(loss_mbce_ori)

                # 3. MBCE Loss on fake features
                loss_mbce_fake = torch.tensor(0.0, device=device)
                weight_fake = 0
                if logits_fake is not None and logits_fake.numel() > 0:
                    fake_labels = torch.zeros(
                        (logits_fake.shape[0], logits_fake.shape[2], 1),
                        dtype=torch.long, device=device
                    )
                    logits_fake_new = logits_fake[:, n_old_classes:, :, :, :]
                    
                    if logits_fake_new.numel() > 0:
                        loss_mbce_fake = self.bce_fake_criterion(
                            logits_fake_new, fake_labels
                        ).mean(dim=[0, 2, 3, 4])
                        
                        if torch.isnan(loss_mbce_fake).any() or torch.isinf(loss_mbce_fake).any():
                            logger.warning(f"NaN in loss_mbce_fake, setting to 0")
                            loss_mbce_fake = torch.zeros_like(loss_mbce_fake)
                        else:
                            stride_num = features[-1].shape[0] * features[-1].shape[2] * features[-1].shape[3] * features[-1].shape[4]
                            weight_fake = fake_labels.shape[1] / (stride_num + 1e-6)

                # 4. MBCE Loss on extra background regions
                loss_mbce_extra_bg = torch.tensor(0.0, device=device)
                weight_extra_bg = 0
                if logits_extra_bg is not None and logits_extra_bg.numel() > 0:
                    extra_bg_labels = torch.zeros(
                        (logits_extra_bg.shape[0], logits_extra_bg.shape[2], 1),
                        dtype=torch.long, device=device
                    )
                    logits_extra_bg_new = logits_extra_bg[:, n_old_classes:, :, :, :]
                    
                    if logits_extra_bg_new.numel() > 0:
                        loss_mbce_extra_bg = self.bce_extra_bg_criterion(
                            logits_extra_bg_new, extra_bg_labels
                        ).mean(dim=[0, 2, 3, 4])
                        
                        if torch.isnan(loss_mbce_extra_bg).any() or torch.isinf(loss_mbce_extra_bg).any():
                            logger.warning(f"NaN in loss_mbce_extra_bg, setting to 0")
                            loss_mbce_extra_bg = torch.zeros_like(loss_mbce_extra_bg)
                        else:
                            stride_num = features[-1].shape[0] * features[-1].shape[2] * features[-1].shape[3] * features[-1].shape[4]
                            weight_extra_bg = region_bg.sum() / (stride_num + 1e-6)

                # Combined MBCE - check denominator
                denom = 1 + weight_fake + weight_extra_bg
                if denom < 1e-6:
                    denom = 1.0
                loss_mbce = (loss_mbce_ori + loss_mbce_fake * weight_fake + loss_mbce_extra_bg * weight_extra_bg) / denom
                loss_mbce = loss_mbce.sum()

                if torch.isnan(loss_mbce) or torch.isinf(loss_mbce):
                    logger.warning(f"NaN in combined loss_mbce, setting to 0")
                    loss_mbce = torch.tensor(0.0, device=device)

                # 5. PKD Loss
                loss_pkd = torch.tensor(0.0, device=device)
                if self.pkd_criterion is not None:
                    loss_pkd = self.pkd_criterion(features, features_old, pseudo_label_region)
                    if torch.isnan(loss_pkd) or torch.isinf(loss_pkd):
                        logger.warning(f"NaN in loss_pkd, setting to 0")
                        loss_pkd = torch.tensor(0.0, device=device)

                # 6. Uncertainty-Aware Constraint (UAC) Loss
                loss_uncertainty = torch.tensor(0.0, device=device)
                if self.uncertainty_criterion is not None:
                    uncer_map = calculate_certainty(torch.sigmoid(logits))
                    
                    fg = ((label_batch == logits.argmax(dim=1)) | 
                        (torch.max(torch.sigmoid(logits), dim=1)[0] > 0.7))
                    mask_fg = torch.where(fg, torch.zeros_like(uncer_map.squeeze(1)),
                                        torch.ones_like(uncer_map.squeeze(1))).float()
                    
                    loss_uncertainty = self.uncertainty_criterion(
                        (1 - uncer_map.squeeze(1)) * mask_fg,
                        torch.zeros_like(uncer_map.squeeze(1))
                    )
                    
                    if torch.isnan(loss_uncertainty) or torch.isinf(loss_uncertainty):
                        logger.warning(f"NaN in loss_uncertainty, setting to 0")
                        loss_uncertainty = torch.tensor(0.0, device=device)

                # 7. Contrastive Prototype Discriminative (CPD) Loss
                loss_cont = torch.tensor(0.0, device=device)
                if self.cont_criterion is not None and self.prev_prototypes is not None:
                    loss_cont = self.cont_criterion(
                        features[-1], logits, label_batch, n_new_classes, self.prev_prototypes
                    )
                    if torch.isnan(loss_cont) or torch.isinf(loss_cont):
                        logger.warning(f"NaN in loss_cont, setting to 0")
                        loss_cont = torch.tensor(0.0, device=device)

                # Combined Loss
                loss_ce = loss_ce_1
                loss_dice = loss_dice_1

                loss = (0.5 * loss_ce + 0.5 * loss_dice + 
                    0.5 * loss_mbce +
                    self.pkd_weight * loss_pkd + 
                    self.cont_weight * loss_cont +
                    self.uncertainty_weight * loss_uncertainty)

                # Final NaN check with detailed logging
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf in final loss at iteration {iter_num}!")
                    logger.error(f"  CE: {loss_ce.item():.4f}, Dice: {loss_dice.item():.4f}")
                    logger.error(f"  MBCE: {loss_mbce.item() if torch.is_tensor(loss_mbce) else loss_mbce:.4f}")
                    logger.error(f"  PKD: {loss_pkd.item() if torch.is_tensor(loss_pkd) else loss_pkd:.4f}")
                    logger.error(f"  Cont: {loss_cont.item() if torch.is_tensor(loss_cont) else loss_cont:.4f}")
                    logger.error(f"  Unc: {loss_uncertainty.item() if torch.is_tensor(loss_uncertainty) else loss_uncertainty:.4f}")
                    # Use only CE + Dice if NaN
                    loss = 0.5 * loss_ce + 0.5 * loss_dice
                
                loss_tot = loss
                dice = 1 - loss_dice_1

                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce
                
                loss_tot.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                
                # if hasattr(model, 'fc') and hasattr(model.fc, 'grad_update'):
                #     with torch.no_grad():
                #         grad_update = (model.fc.mu.weight.grad.clone().detach()) ** 2
                #         model.fc.grad_update.data = grad_update
                #         del grad_update
                
                optimizer.step()
                
                # Log detailed loss breakdown
                if iter_num % 50 == 0:
                    logger.info(f"  [Loss Breakdown] MBCE: {loss_mbce.item():.4f}, "
                               f"PKD: {loss_pkd if isinstance(loss_pkd, float) else loss_pkd.item():.4f}, "
                               f"CPD: {loss_cont if isinstance(loss_cont, float) else loss_cont.item():.4f}, "
                               f"UAC: {loss_uncertainty if isinstance(loss_uncertainty, float) else loss_uncertainty.item():.4f}")

            # del feat if self.step == 0 else features, logits, image_batch, label_batch
            if self.step==0:
                del feat
            else :
                del features,logits,image_batch,label_batch
            
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch: %d, iteration: %d, train loss: %f, train loss_ce: %f, '
                          'train loss_dice: %f, train dice: %f' % 
                          (cur_epoch, iter_num, loss.item(), loss_ce.item(), 
                           loss_dice.item(), dice.item()))
            
            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)
        
        # ============= Epoch End Processing =============
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
        
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i] / class_freq[i]
        
        logger.info('epoch: %d, mean train loss: %f, mean train ce loss: %f, mean train dice: %f' % 
                   (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch: %d, mean train loss: %f, mean train ce loss: %f, mean train dice: %f' % 
              (cur_epoch, mean_loss, mean_loss_ce, mean_dice))

        # ============= Adaptive Deviation Compensation (ADC) =============
        # Refine prototypes periodically
        if self.step > 0 and (cur_epoch + 1) % 10 == 0 and prev_protos is not None:
            logger.info("\n" + "=" * 80)
            logger.info(f"RUNNING ADC AT EPOCH {cur_epoch + 1}")
            self._refine_prototypes_adc(train_loader)
            logger.info("=" * 80 + "\n")

        # Save checkpoints
        save_interval = 50
        if (cur_epoch + 1) >= 350 and (cur_epoch + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(cur_epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))

        if cur_epoch + 1 == max_epoch:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(cur_epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))

        return class_dice, mean_dice

    def validate(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, 
                 novel=False, cur_epoch=None, snapshot_path=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        logger = self.logger

        model.eval()
        iter_num = cur_epoch * len(loader)
        num_classes = self.opts.num_classes
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                
                outputs = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    outputs, label_batch, self.total_classes, softmax=True)

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                del  outputs, image_batch, label_batch
                iter_num += 1

                if iter_num % 10 == 0:
                    logger.info('epoch: %d, iteration: %d, val loss: %f, val loss_ce: %f, '
                              'val loss_dice: %f, val dice: %f' % 
                              (cur_epoch, iter_num, loss.item(), loss_ce.item(), 
                               loss_dice.item(), dice.item()))
                    print('epoch: %d, iteration: %d, val loss: %f, val loss_ce: %f, '
                          'val loss_dice: %f, val dice: %f' % 
                          (cur_epoch, iter_num, loss.item(), loss_ce.item(), 
                           loss_dice.item(), dice.item()))

        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
                
        logger.info('epoch: %d, mean val loss: %f, mean val ce loss: %f, mean val dice: %f' % 
                   (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch: %d, mean val loss: %f, mean val ce loss: %f, mean val dice: %f' % 
              (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def test(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, 
             novel=False, cur_epoch=None, snapshot_path=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        logger = self.logger

        model.eval()
        iter_num = cur_epoch * len(loader)
        num_classes = self.opts.num_classes
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                
                outputs = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    outputs, label_batch, self.total_classes, softmax=True)

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                del outputs, image_batch, label_batch
                
                if iter_num % 5 == 0:
                    print('iteration: %d, test loss: %f, test loss_ce: %f, '
                          'test loss_dice: %f, test dice: %f' % 
                          (iter_num, loss.item(), loss_ce.item(), 
                           loss_dice.item(), dice.item()))

                iter_num += 1

        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))
        
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]

        logger.info('epoch: %d, mean test loss: %f, mean test ce loss: %f, mean test dice: %f' % 
                   (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('Mean Test loss: %f, mean Test ce loss: %f, Mean Test dice: %f' % 
              (mean_loss, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def state_dict(self):
        state = {"model": self.model.state_dict(), 
                 "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state
        
    def load_body(self, model_dict):
        new_state = {}
        for k, v in model_dict.items():
            if "outc" not in k:
                new_state[k] = v
        return new_state
    
    def load_dict_imprint(self, path, strict=True):
        print("Inside load dict imprint")
        if self.opts.step == 1:
            nc = 16
        elif self.opts.step == 2:
            nc = 21
        elif self.opts.step == 3:
            nc = 27
        elif self.opts.step == 4:
            nc = 31
        elif self.opts.step == 5:
            nc = 34
        elif self.opts.step == 6:
            nc = 38
            
        # for name, param in path.items():
        #     if name.startswith('fc'):
        #         print(self.model.state_dict()[name][0:nc].shape, param.shape)
        #         self.model.state_dict()[name][0:nc].copy_(param)
        #         continue
        #     if name in self.model.state_dict():
        #         self.model.state_dict()[name].copy_(param)
        
        for name, param in self.model.named_parameters():
            if name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = True

    def load_dict(self, path, strict=True):
        body = self.load_body(path)
        self.model.load_state_dict(body, strict)
    
    def load_dict_full_model(self, path, strict=True):
        self.model.load_state_dict(path, strict)

    def save_protos(self, dataset, step):
        """Save prototypes with enhanced statistics for ADC"""
        protos = {}
        proto_stats = {}
        
        # Load existing prototypes if continuing from previous step
        if step > 0:
            proto_path = '../../../base/codu_run/codu/saved_proto.pkl' if step == 1 else 'saved_proto.pkl'
            if os.path.exists(proto_path):
                with open(proto_path, 'rb') as f:
                    protos = pickle.load(f)

        model = self.model.module if self.distributed else self.model
        model.eval()

        # Determine which classes to save prototypes for
        classes = []
        if step == 0:
            classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif step == 1:
            classes = [16, 17, 18, 19, 20]
        elif step == 2:
            classes = [21, 22, 23, 24, 25, 26]
        elif step == 3:
            classes = [27, 28, 29, 30]
        elif step == 4:
            classes = [31, 32, 33]
        elif step == 5:
            classes = [34, 35, 36, 37]
        
        self.logger.info("=" * 80)
        self.logger.info(f"SAVING PROTOTYPES FOR STEP {step}")
        self.logger.info(f"Classes: {classes}")
        
        sam = 10 if step == 0 else self.task.nshot

        for c in classes:
            self.logger.info(f"\nProcessing class {c}...")
            
            ds = dataset.get_k_image_of_class(cl=c, k=sam)
            wc, bg = get_prototype(model, ds, c, self.device, 
                                  interpolate_label=False, return_all=False, background=True)
            
            if wc is not None:
                wc = wc.view(-1, 1, 1, 1)
                bg = bg.view(-1, 1, 1, 1)
                
                protos[c] = wc
                bgkey = 'bg' + str(c)
                protos[bgkey] = bg
                
                # Compute and store statistics for ADC
                proto_norm = torch.norm(wc.squeeze(), p=2).item()
                proto_std = wc.squeeze().std().item()
                
                proto_stats[c] = {
                    'norm_mean': proto_norm,
                    'norm_std': proto_norm * 0.1,
                    'noise_std': proto_std
                }
                
                self.logger.info(f"  Class {c} prototype saved")
                self.logger.info(f"    Shape: {wc.shape}")
                self.logger.info(f"    Norm: {proto_norm:.4f}")
                self.logger.info(f"    Std: {proto_std:.4f}")
            else:
                raise Exception(f"Unable to save prototype for class {c}")

        # Save prototypes
        with open('saved_proto.pkl', 'wb') as f:
            pickle.dump(protos, f)
        
        # Save statistics separately for easy access
        with open('saved_proto_stats.pkl', 'wb') as f:
            pickle.dump(proto_stats, f)
        
        self.logger.info(f"\nTotal prototypes saved: {len([k for k in protos.keys() if isinstance(k, int)])}")
        self.logger.info("Prototypes saved to: saved_proto.pkl")
        self.logger.info("Statistics saved to: saved_proto_stats.pkl")
        self.logger.info("=" * 80)