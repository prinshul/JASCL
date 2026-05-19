import torch
from torch import distributed
import torch.nn as nn
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
import torch.functional as F
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
class PKDLoss(nn.Module):
    """Pixel-wise Knowledge Distillation Loss"""
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
            
            # Compute cosine similarity
            similarity = (feat_new_norm * feat_old_norm).sum(dim=1, keepdim=True)
            
            # Only compute loss on pseudo-labeled regions
            masked_loss = ((1 - similarity) * pseudo_label_region).sum() / (pseudo_label_region.sum() + 1e-6)
            loss += masked_loss
            
            if self.call_count % 100 == 0:
                print(f"    PKD layer {idx}: similarity_mean={similarity.mean().item():.4f}, masked_loss={masked_loss.item():.4f}")
        
        final_loss = loss / len(features_new)
        
        if self.call_count % 100 == 0:
            print(f"    PKD total loss: {final_loss.item():.4f} (averaged over {len(features_new)} layers)")
        
        return final_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss to separate new classes from old prototypes"""
    def __init__(self, n_old_classes, n_new_classes):
        super().__init__()
        self.n_old_classes = n_old_classes
        self.n_new_classes = n_new_classes
        self.call_count = 0
    
    def forward(self, features, logits_new, labels, old_prototypes):
        """
        features: [N, C, D, H, W] - feature map from encoder
        logits_new: [N, n_new_classes, D, H, W] - logits for new classes
        labels: [N, D, H, W] - ground truth labels
        old_prototypes: [n_old_classes, C] - prototypes of old classes
        """
        self.call_count += 1
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)  # [N, C, D, H, W]
        old_prototypes_norm = F.normalize(old_prototypes, p=2, dim=1)  # [n_old, C]
        
        loss = 0
        count = 0
        
        # For each new class in the current labels
        unique_labels = labels.unique()
        
        if self.call_count % 100 == 0:
            print(f"    Contrastive: processing {len(unique_labels)} unique labels: {unique_labels.cpu().numpy()}")
        
        for cls_idx in unique_labels:
            if cls_idx == 0 or cls_idx == 255:  # Skip background and ignore index
                continue
            
            # Get mask for current class
            mask = (labels == cls_idx)  # [N, D, H, W]
            if mask.sum() == 0:
                continue
            
            # Get features for this class
            mask_expanded = mask.unsqueeze(1)  # [N, 1, D, H, W]
            
            # Extract features for this class and flatten
            class_features = features_norm * mask_expanded.float()
            class_features = class_features.permute(1, 0, 2, 3, 4)  # [C, N, D, H, W]
            class_features = class_features.reshape(class_features.shape[0], -1)  # [C, N*D*H*W]
            class_features = class_features[:, mask.flatten()]  # [C, num_pixels]
            
            # Compute similarity with all old prototypes
            similarities = torch.mm(old_prototypes_norm, class_features)  # [n_old, num_pixels]
            
            # We want to minimize similarity (maximize dissimilarity)
            margin = 0.5
            class_loss = F.relu(similarities - margin).mean()
            loss += class_loss
            count += 1
            
            if self.call_count % 100 == 0:
                print(f"      Class {cls_idx.item()}: {mask.sum().item()} pixels, max_similarity={similarities.max().item():.4f}, loss={class_loss.item():.4f}")
        
        final_loss = loss / max(count, 1)
        
        if self.call_count % 100 == 0:
            print(f"    Contrastive total loss: {final_loss.item():.4f} (averaged over {count} classes)")
        
        return final_loss

class Trainer:
    def __init__(self, task, device, logger, opts, config_vit):
        self.logger = logger
        self.device = device
        self.c1=0.7
        self.c2=0.3
        self.task = task
        self.opts = opts
        self.novel_classes = self.task.get_n_classes()[-1]
        self.labels_old = task.get_old_labels(bkg=False)
       
        self.step = task.step        
        self.labels = self.task.get_novel_labels()
        self.total_classes = self.labels + self.labels_old
        self.need_model_old = (self.opts.born_again or self.opts.mib_kd > 0 or self.opts.loss_kd > 0 or
                               self.opts.l2_loss > 0 or self.opts.l1_loss > 0 or self.opts.cos_loss > 0)

        self.n_channels = -1  # features size, will be initialized in make model
        self.model = self.make_model(config_vit)
        self.model = self.model.to(device)
        self.svf = opts.svf
        self.distributed = False
        self.model_old = None

        if self.opts.fix_bn:
            self.model.fix_bn()

        if self.opts.bn_momentum is not None:
            self.model.bn_set_momentum(self.opts.bn_momentum)

        self.initialize(self.opts)  # initialize model parameters (e.g. perform WI)

        self.born_again = self.opts.born_again
        self.dist_warm_start = self.opts.dist_warm_start
        model_old_as_new = self.opts.born_again or self.opts.dist_warm_start
        if self.need_model_old:
            self.model_old = self.make_model(config_vit, is_old=not model_old_as_new)
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.to(device)
            self.model_old.eval()

        # xxx Set up optimizer
        self.train_only_novel = self.opts.train_only_novel

        # self.optimizer = torch.optim.SGD(params, lr=self.opts.lr, momentum=0.9, weight_decay=self.opts.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 25, eta_min=5e-6)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)
        if self.step > 0 and self.model_old is not None:
            self.pkd_criterion = PKDLoss()
            self.cont_criterion = ContrastiveLoss(
                n_old_classes=len(self.labels_old),
                n_new_classes=len(self.labels)
            )
            self.pkd_weight = opts.pkd_weight if hasattr(opts, 'pkd_weight') else 1.0
            self.cont_weight = opts.cont_weight if hasattr(opts, 'cont_weight') else 0.1
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
            
    def make_model(self, config_vit, is_old=False):
        # classifier, self.n_channels = self.get_classifier(is_old)
        n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
        model = make_model(self.opts, config_vit, n_classes[0])
        return model

    def distribute(self):
        self.opts = self.opts
        if self.model is not None:
            # Put the model on GPU
            self.distributed = True
            self.model = DistributedDataParallel(self.model, device_ids=[self.opts.device_id],
                                                 output_device=self.opts.device_id, find_unused_parameters=True)

    def get_classifier(self, is_old=False):
        # here distinguish methods!
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
        """Compute statistics for each prototype (mean norm and std)"""
        stats = {}
        
        self.logger.info("-" * 80)
        self.logger.info("COMPUTING PROTOTYPE STATISTICS")
        self.logger.info("-" * 80)
        
        for cls_idx, proto in protos.items():
            if not isinstance(cls_idx, int):
                continue
            
            # proto shape: [C, 1, 1, 1] or [C]
            if proto.dim() == 4:
                proto = proto.squeeze(-1).squeeze(-1).squeeze(-1)
            
            # Compute norm statistics
            norm_mean = torch.norm(proto, p=2).item()
            norm_std = norm_mean * 0.1  # Simplified: assume 10% std
            
            # Compute noise
            noise_std = proto.std().item()
            
            stats[cls_idx] = {
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'noise_std': noise_std
            }
            
            self.logger.info(f"Class {cls_idx}: norm_mean={norm_mean:.4f}, norm_std={norm_std:.4f}, noise_std={noise_std:.4f}")
        
        self.logger.info(f"Total prototypes with statistics: {len(stats)}")
        self.logger.info("-" * 80)
        
        return stats

    def _generate_fake_features(self, protos, proto_stats, samples_per_class):
        """Generate fake features from prototypes with noise"""
        fake_features_list = []
        
        self.logger.debug(f"Generating fake features: {samples_per_class} samples per class")
        
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
            per_cls_fake_features = proto.repeat(1, 1, samples_per_class, 1)  # [C, 1, N, 1]
            
            # Add noise based on statistics
            if proto_stats is not None and cls_idx in proto_stats:
                noise_std = proto_stats[cls_idx]['noise_std']
                noise = torch.randn_like(per_cls_fake_features) * noise_std
                per_cls_fake_features = per_cls_fake_features + noise
                
                # Scale by norm statistics
                norm_mean = proto_stats[cls_idx]['norm_mean']
                norm_std = proto_stats[cls_idx]['norm_std']
                rand_norm = torch.randn(1, 1, samples_per_class, 1).to(proto.device) * norm_std + norm_mean
                per_cls_fake_features = F.normalize(per_cls_fake_features, p=2, dim=0) * rand_norm
                
                self.logger.debug(f"  Class {cls_idx}: generated {samples_per_class} samples with shape {per_cls_fake_features.shape}")
            
            fake_features_list.append(per_cls_fake_features)
        
        # Concatenate all fake features
        if len(fake_features_list) > 0:
            fake_features = torch.cat(fake_features_list, dim=2)  # [C, 1, total_samples, 1]
            fake_features = fake_features.to(self.device)
            
            self.logger.debug(f"Total fake features generated: {fake_features.shape}")
            
            return fake_features
        
        self.logger.warning("No fake features generated!")
        return None

    def warm_up(self, dataset, epochs=1):
        self.warm_up_(dataset, epochs)
        # warm start means make KD after weight imprinting or similar
        if self.dist_warm_start:
            self.model_old.load_state_dict(self.model.state_dict())

    def warm_up_(self, dataset, epochs=1):
        pass

    def cool_down(self, dataset, epochs=1):
        pass
    
    def visualize_pred(self, labels, prediction, cur_epoch, cur_step):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        
        sitk.WriteImage(label_arr, os.path.join('data/merged/dataset/prediction/ground_truth','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('data/merged/dataset/prediction/predicted_mask','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))

    def visualize_pred_test(self, labels, images, prediction, i_batch):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        image_arr = sitk.GetImageFromArray(images.detach().cpu().numpy())

        sitk.WriteImage(label_arr, os.path.join('data/merged/dataset/prediction/test/ground_truth','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('data/merged/dataset/prediction/test/predicted_mask','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(image_arr, os.path.join('data/merged/dataset/prediction/test/images','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))

    def train(self, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path=None):
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
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.opts.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0

        # Load previous prototypes and compute statistics for fake feature generation
        prev_protos = None
        prev_proto_stats = None
        if self.opts.step > 0:
            print("Previous prototypes loaded.")
            if self.opts.step == 1:
                with open('../../../base/codu_run/codu/saved_proto.pkl', 'rb') as f:
                    prev_protos = pickle.load(f)
            else:
                with open('saved_proto.pkl', 'rb') as f:
                    prev_protos = pickle.load(f)
            
            # Compute prototype statistics (mean norm and noise)
            prev_proto_stats = self._compute_prototype_statistics(prev_protos)
        
        # Compute number of fake samples per iteration (simplified version)
        if self.opts.step > 0 and prev_protos is not None:
            n_old_classes = len([k for k in prev_protos.keys() if isinstance(k, int)])
            samples_per_class_per_iter = max(1, 100 // n_old_classes)  # Adjust as needed
        
        for i_batch, sampled_batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            if self.opts.step == 0:
                # Base step - standard training
                feat, logits = model(image_batch)
                
                loss_ce_1 = ce_loss(logits, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(logits, label_batch[:].long(), self.total_classes, softmax=True)

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
                # Incremental step - use prototype replay
                
                # Generate fake features from old prototypes
                fake_features = self._generate_fake_features(
                    prev_protos, prev_proto_stats, samples_per_class_per_iter
                )
                
                # Get predictions from old model for pseudo-labeling
                with torch.no_grad():
                    _, features_old = self.model_old(image_batch, ret_intermediate=True)
                    logits_old, _ = self.model_old(image_batch)
                    
                    pred_old = logits_old.argmax(dim=1)  # [N, D, H, W]
                    
                    # Create pseudo-label region: where current labels are background but old model predicts a class
                    pseudo_label_region = torch.logical_and(
                        label_batch == 0,
                        pred_old > 0
                    ).unsqueeze(1).float()  # [N, 1, D, H, W]
                
                # Forward pass with fake features
                logits, feat, logits_fake = model(image_batch, fake_features=fake_features, ret_intermediate=True)
                
                # === Compute losses ===
                
                # 1. Standard CE + Dice loss on real data
                L_S3C = ce_loss(logits, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(logits, label_batch[:].long(), self.total_classes, softmax=True)
                
                # 2. Prototype replay loss (fake features should be classified as background/old classes)
                # Create labels for fake features (all background)
                fake_labels = torch.zeros(
                    (fake_features.shape[0], fake_features.shape[2]),
                    dtype=torch.long, device=device
                )  # [1, N_fake_samples]
                fake_labels = fake_labels.unsqueeze(-1)  # Add spatial dims if needed
                
                loss_proto_replay = 0
                if logits_fake is not None:
                    loss_proto_replay = ce_loss(logits_fake.squeeze(-1), fake_labels.squeeze(-1))
                
                # 3. PKD Loss - preserve old model's features
                loss_pkd = 0
                if self.pkd_criterion is not None:
                    # Need to get intermediate features - modify forward pass
                    _, features_new = model(image_batch, ret_intermediate=True)
                    loss_pkd = self.pkd_criterion([features_new], [features_old], pseudo_label_region)
                
                # 4. Contrastive Loss - separate new classes from old prototypes
                loss_cont = 0
                if self.cont_criterion is not None and prev_protos is not None:
                    # Extract old class prototypes
                    old_prototypes_list = []
                    for cls_idx in sorted([k for k in prev_protos.keys() if isinstance(k, int)]):
                        proto = prev_protos[cls_idx]
                        if proto.dim() == 4:  # [C, 1, 1, 1]
                            proto = proto.squeeze(-1).squeeze(-1).squeeze(-1)  # [C]
                        old_prototypes_list.append(proto)
                    
                    if len(old_prototypes_list) > 0:
                        old_prototypes_tensor = torch.stack(old_prototypes_list)  # [n_old, C]
                        
                        # Get logits for new classes only
                        n_old = len(self.labels_old)
                        logits_new = logits[:, n_old:, :, :, :]  # Assuming class dim is 1
                        
                        loss_cont = self.cont_criterion(feat, logits_new, label_batch, old_prototypes_tensor)
                
                # Combined loss
                loss_ce = L_S3C + 0.1 * loss_proto_replay
                loss_dice = loss_dice_1
                
                loss = 0.5 * loss_ce + 0.5 * loss_dice + \
                    self.pkd_weight * loss_pkd + \
                    self.cont_weight * loss_cont
                
                loss_tot = loss
                dice = 1 - loss_dice_1

                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce
                
                loss_tot.backward()
                
                if hasattr(model, 'fc') and hasattr(model.fc, 'grad_update'):
                    with torch.no_grad():
                        grad_update = (model.fc.mu.weight.grad.clone().detach()) ** 2
                        model.fc.grad_update.data = grad_update
                        del grad_update
                
                optimizer.step()

            del feat, logits, image_batch, label_batch
            
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                    cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
            
            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)
        
        # Mean loss and mean dice
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
        
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i] / class_freq[i]
        
        logger.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))

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

    def validate(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, novel=False, cur_epoch=None, snapshot_path = None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        # criterion = self.criterion
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
                # outputs, _ = model(image_batch)
                emb, outputs = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch, self.total_classes, softmax=True)


                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                # _, prediction = outputs.max(dim=1)
                # metrics.update(label_batch[:].long(), prediction)4
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                del emb, outputs, image_batch, label_batch
                iter_num += 1

                if iter_num % 10 == 0:
                    logger.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
                    print('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))

                # if(iter_num % 2 == 0):
                #     res_array = np.zeros(26).tolist()
                #     for i in range(26):
                #         if class_freq[i] != 0:
                #             res_array[i] = class_dice[i]/class_freq[i]
                            
                    # print("val dices:", res_array)
        # mean loss and mean dice
        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))
        # mean_class_dice = list(map(lambda x: x/len(loader) if x != 'X' else x, class_dice))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
                
        logger.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def test(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, novel=False, cur_epoch=None, snapshot_path = None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        # criterion = self.criterion
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
                # outputs, _ = model(image_batch)
                emb, outputs = model(image_batch)
                #print(emb.shape, outputs.shape)
                # print("Ground Truth", torch.unique(label_batch))
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch, self.total_classes, softmax=True)

                # print("Ground Truth", torch.unique(label_batch))
                # print("Prediction", torch.unique(outputs))

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                # _, prediction = outputs.max(dim=1)
                # metrics.update(label_batch[:].long(), prediction)

                # _, prediction = outputs.max(dim=1)
                
                # print("Prediction", torch.unique(prediction))
                    
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                #self.visualize_pred_test(label_batch, image_batch, outputs, i_batch)           
                del emb, outputs, image_batch, label_batch
                
                if iter_num % 5 == 0:
                    print('iteration : %d, test loss : %f, test loss_ce: %f, test loss_dice: %f, test dice : %f' % (
                        iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))


                iter_num += 1

        # mean loss and mean dice
        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))
        
        # print(class_dice)
        # print(class_freq)
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]

        # mean_class_dice = list(map(lambda x: x/len(loader) if x != 'X' else x, class_dice))

        logger.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('Mean Test loss : %f, mean Test ce loss: %f, Mean Test dice : %f' % (mean_loss, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state
        
    def load_body(self, model_dict):
        new_state = {}
        for k, v in model_dict.items():
            # if "hybrid_model" in k or "encoder1" in k:
            if "outc" not in k:
                new_state[k] = v
        return new_state
    

    def load_dict_imprint(self, path, strict=True):
        print("Inside load dict imprint")
        if self.opts.step==1:
            nc=16
        elif self.opts.step==2:
            nc=21
        elif self.opts.step==3:
            nc=27
        elif self.opts.step==4:
            nc=31
        elif self.opts.step==5:
            nc=34
        elif self.opts.step==6:
            nc=38
            
        for name, param in path.items():
            if name.startswith('fc'):
                print(self.model.state_dict()[name][0:nc].shape, param.shape)
                self.model.state_dict()[name][0:nc].copy_(param)
                continue
            if name in self.model.state_dict():
                self.model.state_dict()[name].copy_(param)

        # Next, randomize the parameters of the final layer for the additional columns
        # with torch.no_grad():
        #     self.model.outc.weight[:len(self.labels_old) + 1, :] = path['outc.weight']            
        # for name, param in path.items():
        #     if name.startswith('outc'):
        #         self.model.state_dict()[name].requires_grad = True
        #     else:
        #         self.model.state_dict()[name].requires_grad = False
        
        #if self.svf:
            #self.model.fc.sigma = resolver(self.model.fc.sigma, global_low_rank_ratio=1.0, skip_1x1=False, skip_3x3=False).to(self.device)
        
        
        # for param in self.model.parameters():
        #     param.requires_grad = True
        
        for name, param in self.model.named_parameters():
            if name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
            
            
        # for param in self.model.parameters():
        #     print(param.requires_grad)
        # for name, param in path.items():
        #     print(name)
        #     print(self.model.state_dict()[name].requires_grad)
        #     # param.requires_grad = True

    def load_dict(self, path, strict=True):
        body = self.load_body(path)
        self.model.load_state_dict(body, strict)
    
    def load_dict_full_model(self, path, strict=True):
        self.model.load_state_dict(path, strict)
        


    def save_protos(self, dataset, step):
        protos={}
        if step>0:
            if step==1:
                with open('../../../base/codu_run/codu/saved_proto.pkl', 'rb') as f:
                    protos=pickle.load(f)
            else:
                with open('saved_proto.pkl', 'rb') as f:
                    protos=pickle.load(f)

        model = self.model.module if self.distributed else self.model
        
        model.eval()

        classes=[]
        if step==0:
            classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif step==1:
            classes=[16, 17, 18, 19, 20]
        elif step==2:
            classes=[21, 22, 23, 24, 25, 26]
        elif step==3:
            classes=[27, 28, 29, 30]
        elif step==4:
            classes=[31,32,33]
        elif step==5:
            classes=[34,35,36,37]
        
        print("Saving protos for step {}, classes {}".format(step, classes))
        sam=0

        if step==0:
            sam=10
        else:
            sam=self.task.nshot

        for c in classes:
            weight = None
            count = 0
        
            ds = dataset.get_k_image_of_class(cl=c, k=sam)
            print("#########Good here")
            wc, bg = get_prototype(model, ds, c, self.device, interpolate_label=False, return_all=False, background=True)
            print("#########Good here")
            wc=wc.view(-1, 1, 1, 1)
            bg=bg.view(-1, 1, 1, 1)
            print(wc.shape, bg.shape)
            if wc is not None:
                #wc = wc.unsqueeze(3)
                protos[c]=wc
                bgkey='bg'+str(c)
                protos[bgkey]=bg
            else:
                raise Exception(f"Unable to imprint weight of class {c} after {count} trials.")
            print("Prototype for class {} saved".format(c))

        for key, val in protos.items():
            print(key, val.shape)
      
        with open('saved_proto.pkl', 'wb') as f:
            pickle.dump(protos, f)
