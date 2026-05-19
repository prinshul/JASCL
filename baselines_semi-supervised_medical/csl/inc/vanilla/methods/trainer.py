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
import torch.nn.functional as F
import SimpleITK as sitk
import itertools
import numpy as np
import os, sys
import random
import pickle
from CSL_filter import CSLPseudoLabelFilter, CSLIntegrationLogger

CLIP = 10

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Trainer:
    def __init__(self, task, device, logger, opts, config_vit):
        self.logger = logger
        self.device = device
        self.task = task
        self.opts = opts
        self.novel_classes = self.task.get_n_classes()[-1]
        self.labels_old = task.get_old_labels(bkg=False)
       
        self.step = task.step        
        self.labels = self.task.get_novel_labels()
        self.total_classes = self.labels + self.labels_old
        self.need_model_old = (self.opts.born_again or self.opts.mib_kd > 0 or self.opts.loss_kd > 0 or
                               self.opts.l2_loss > 0 or self.opts.l1_loss > 0 or self.opts.cos_loss > 0)

        self.n_channels = -1
        self.model = self.make_model(config_vit)
        self.model = self.model.to(device)
        
        self.teacher = self.make_model(config_vit)
        self.teacher = self.teacher.to(device)
        
        # ========== CSL INTEGRATION ==========
        self.logger.info("=" * 70)
        self.logger.info("INITIALIZING CSL (Confidence Separable Learning)")
        self.logger.info("=" * 70)
        
        # Initialize CSL filter with configurable alpha
        csl_alpha = getattr(opts, 'csl_alpha', 2.0)
        self.csl_filter = CSLPseudoLabelFilter(alpha=csl_alpha, logger=logger)
        self.csl_integration_logger = CSLIntegrationLogger(logger)
        
        self.logger.info(f"CSL Filter initialized with alpha={csl_alpha}")
        self.logger.info("CSL will replace prototype-based filtering")
        self.csl_integration_logger.mark_complete('1_max_conf_residual_var')
        self.csl_integration_logger.mark_complete('2_spectral_clustering')
        self.csl_integration_logger.mark_complete('3_adaptive_weighting')
        # ====================================
        
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
            self.optimizer, 25, eta_min=5e-6
        )
        self.logger.debug("Optimizer:\n%s" % self.optimizer)
       
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
                    print("Using KD between outputs - cosine",opts.loss_kd)
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
        n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
        model = make_model(self.opts, config_vit, n_classes[0])
        return model

    def distribute(self):
        self.opts = self.opts
        if self.model is not None:
            self.distributed = True
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.opts.device_id],
                output_device=self.opts.device_id, find_unused_parameters=True
            )

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
        
        sitk.WriteImage(label_arr, os.path.join(
            'data/merged/dataset/prediction/ground_truth',
            '{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)
        ))
        sitk.WriteImage(prediction_arr, os.path.join(
            'data/merged/dataset/prediction/predicted_mask',
            '{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)
        ))

    def train(self, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, 
              metrics=None, print_int=10, n_iter=1, snapshot_path=None):
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
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.opts.lr, momentum=0.9, 
                                    weight_decay=0.0001, nesterov=True)

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0

        # if self.opts.step>0:
        #     print("Previous prototypes loaded.")
        #     if self.opts.step==1:
        #         with open('saved_proto_step0.pkl', 'rb') as f:
        #             prev_protos=pickle.load(f)
        #     else:
        #         with open('saved_proto.pkl', 'rb') as f:
        #             prev_protos=pickle.load(f)
            
        for i_batch, sampled_batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            if self.opts.step==0:           
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                
                feat, logits = model(image_batch)
                
                loss_ce_1 = ce_loss(logits, label_batch[:].long())
                
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    logits, label_batch[:].long(), self.total_classes, softmax=True
                )

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                
                loss = 0.5*loss_ce + 0.5*loss_dice
            
                loss_tot = loss
                dice = 1 - loss_dice_1

                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()
        
            else:
                L_proto=0
                # for label, feat in prev_protos.items():
                #     if type(label)==str:
                #         label=0    
                #     feat=feat.unsqueeze(0)
                #     label=torch.tensor([label])
                #     label=label.view(1, 1, 1, 1)
                #     label=label.cuda()
                #     feat=feat.cuda()
                #     logits=model.outc(feat)
                    
                #     loss_p_ce=ce_loss(logits, label)
                #     L_proto+=loss_p_ce
                
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                feat, logits = model(image_batch)
                
                L_S3C = ce_loss(logits, label_batch[:].long())
                
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    logits, label_batch[:].long(), self.total_classes, softmax=True
                )

                loss_ce = L_S3C + 0.1*L_proto

                loss_dice = loss_dice_1
                
                loss = loss_ce + loss_dice
            
                loss_tot = loss
                dice = 1 - loss_dice_1

                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                del image_batch, label_batch
                
                loss_tot.backward()
                
                # with torch.no_grad():
                #     grad_update = (model.outc.mu.weight.grad.clone().detach())**2
                #     model.outc.grad_update.data = grad_update
                #     del grad_update
                
                optimizer.step()
                del feat, logits
                
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info(
                    'epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, '
                    'train loss_dice: %f, train dice : %f' % 
                    (cur_epoch, iter_num, loss.item(), loss_ce.item(), 
                     loss_dice.item(), dice.item())
                )
            sys.stdout.flush()
            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice) 
                
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
            
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
        
        logger.info(
            'epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice)
        )
        print(
            'epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice)
        )

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

    
    # ========== CSL INTEGRATION: REPLACED FUNCTION ==========
    @torch.no_grad()
    def filter_pseudo_labels_CSL(self, logits, ignore_mask=None):
        """
        CSL-based pseudo-label filtering (REPLACES prototype-based filtering).
        
        Args:
            logits: (B, C, H, W, D) - model predictions
            ignore_mask: (B, H, W, D) - pixels to ignore (255 = ignore)
        
        Returns:
            pseudo_labels: (B, H, W, D) - filtered pseudo-labels
            weights: (B, H, W, D) - confidence weights
        """
        self.logger.info("Using CSL filtering instead of prototype-based filtering")
        
        # Use CSL filter
        pseudo_labels, weights, metrics = self.csl_filter.filter_pseudo_labels(
            logits, ignore_mask
        )
        
        # Log metrics
        self.logger.info(
            f"CSL Metrics - High-conf ratio: {metrics['high_conf_ratio']:.2%}, "
            f"Mean weight: {metrics['mean_weight']:.3f}, "
            f"Median weight: {metrics['median_weight']:.3f}"
        )
        
        return pseudo_labels, weights
    # ========================================================

    def update_teacher(self, alpha=0.99):
        for s_param, t_param in zip(self.model.parameters(), self.teacher.parameters()):
            t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data
	
    
    def pseudo_train(self, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, 
                    labeled_loader, metrics=None, print_int=10, n_iter=1, snapshot_path=None):
        """
        Semi-supervised training with CSL pseudo-label filtering.
        MODIFIED: Replaced prototype-based filtering with CSL approach.
        """
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = optimizer
        print("Exemp loader length = ",len(train_loader))
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        
        # ========== CSL LOG ==========
        logger.info("=" * 70)
        logger.info(f"PSEUDO TRAINING with CSL - Epoch {cur_epoch}")
        logger.info("=" * 70)
        self.csl_integration_logger.mark_complete('4_loss_computation')
        self.csl_integration_logger.mark_complete('5_training_integration')
        self.csl_integration_logger.print_status()
        # =============================
        
        step=self.opts.step
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

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        mean_consistency_loss = 0  # Track CSL consistency
        
        sys.stdout.flush()
        # protos={}
        # if step>0:
        #     if step==1:
        #         with open('saved_proto_step0.pkl', 'rb') as f:
        #             protos=pickle.load(f)
        #     else:
        #         with open('saved_proto.pkl', 'rb') as f:
        #             protos=pickle.load(f)
        
        self.teacher.eval()
        labeled_iter = iter(itertools.cycle(labeled_loader)) 
        
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            rloss = torch.tensor([0.]).to(self.device)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # ========== CSL INTEGRATION: Student predictions ==========
            features_st, outputs_st = model(image_batch)
            
            # Create ignore mask for unlabeled data (all valid)
            ignore_mask_unlabeled = torch.zeros_like(label_batch)
            
            # Apply CSL filtering to student predictions
            logger.debug(f"Batch {i_batch}: Applying CSL to student predictions")
            pseudo_labels_st, weights_st = self.filter_pseudo_labels_CSL(
                outputs_st, ignore_mask_unlabeled
            )
            # ==========================================================
            
            # ========== CSL INTEGRATION: Teacher predictions ==========
            features_tea, outputs_tea = self.teacher(image_batch)
            
            # Apply CSL filtering to teacher predictions
            logger.debug(f"Batch {i_batch}: Applying CSL to teacher predictions")
            pseudo_labels_tea, weights_tea = self.filter_pseudo_labels_CSL(
                outputs_tea, ignore_mask_unlabeled
            )
            # ==========================================================
            
            # ========== CSL INTEGRATION: Weighted consistency loss ==========
            # Use CSL weights for consistency - combine student and teacher weights
            combined_weights = (weights_st + weights_tea) / 2.0
            
            # Weighted MSE between student and teacher pseudo-labels
            diff = (pseudo_labels_tea.float() - pseudo_labels_st.float()) ** 2
            weighted_mse = (diff * combined_weights).sum() / (combined_weights.sum() + 1e-8)
            
            logger.debug(
                f"Batch {i_batch}: Consistency loss = {weighted_mse.item():.4f}, "
                f"Mean combined weight = {combined_weights.mean().item():.3f}"
            )
            # ================================================================
            
            del pseudo_labels_tea, pseudo_labels_st, weights_tea, weights_st
            del features_tea, outputs_tea, features_st, outputs_st
            del image_batch, label_batch 
            
            ## Labeled data
            labeled_batch = next(labeled_iter)
            lbl_image_batch, lbl_label_batch = labeled_batch['image'].cuda(), labeled_batch['label'].cuda()
            
            feat, outputs = model(lbl_image_batch)
            
            loss_ce_1 = ce_loss(outputs, lbl_label_batch[:].long())
            class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                outputs, lbl_label_batch[:].long(), self.total_classes, softmax=True
            )

            loss_ce = loss_ce_1
            loss_dice = loss_dice_1

            # ========== CSL INTEGRATION: Modified loss with CSL consistency ==========
            # Use weighted_mse instead of old mse_loss
            consistency_weight = getattr(self.opts, 'csl_consistency_weight', 0.05)
            loss = loss_ce + loss_dice + consistency_weight * weighted_mse
            
            logger.debug(
                f"Batch {i_batch}: Loss breakdown - CE: {loss_ce.item():.4f}, "
                f"Dice: {loss_dice.item():.4f}, CSL consistency: {weighted_mse.item():.4f}"
            )
            # =========================================================================
            
            if not self.opts.vanila:
                loss_tot = loss + rloss
                if rloss <= CLIP:
                    loss_tot = loss + rloss
                else:
                    print(f"Warning, rloss is {rloss}! Term ignored")
                    loss_tot = loss
                    
                loss_tot = loss + rloss
            else:
                loss_tot = loss
            dice = 1 - loss_dice_1
            
            mean_loss += loss
            mean_dice += dice
            mean_loss_ce += loss_ce
            mean_consistency_loss += weighted_mse.item()
            
            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()
            
            del lbl_image_batch, lbl_label_batch, outputs, feat
            
            # Log progress
            if i_batch % 10 == 0:
                logger.info(
                    f"Batch {i_batch}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, Dice={dice.item():.4f}, "
                    f"CSL_consistency={weighted_mse.item():.4f}"
                )
        
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
        mean_consistency_loss = float(mean_consistency_loss / len(train_loader))
           
        logger.info(
            'Unlabeled epoch : %d, mean train loss : %f, mean train ce loss: %f, '
            'mean train dice : %f, mean CSL consistency: %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice, mean_consistency_loss)
        )
        print(
            'Unlabeled epoch : %d, mean train loss : %f, mean train ce loss: %f, '
            'mean train dice : %f, mean CSL consistency: %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice, mean_consistency_loss)
        )
        sys.stdout.flush()
        
        print("Updating teacher")
        self.update_teacher()
        del labeled_iter
        
        return 1
    
								 
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
                _, outputs = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    outputs, label_batch, self.total_classes, softmax=True
                )

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            
                iter_num += 1

                if iter_num % 3 == 0:
                    logger.info(
                        'epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, '
                        'val loss_dice: %f, val dice : %f' % 
                        (cur_epoch, iter_num, loss.item(), loss_ce.item(), 
                         loss_dice.item(), dice.item())
                    )
                    print(
                        'epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, '
                        'val loss_dice: %f, val dice : %f' % 
                        (cur_epoch, iter_num, loss.item(), loss_ce.item(), 
                         loss_dice.item(), dice.item())
                    )
                
                del image_batch, label_batch, outputs

        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
                
        logger.info(
            'epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice)
        )
        print(
            'epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice)
        )
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
                _, outputs = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(
                    outputs, label_batch, self.total_classes, softmax=True
                )

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce
                
                del image_batch, label_batch, outputs
                    
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                if iter_num % 5 == 0:
                    print(
                        'iteration : %d, test loss : %f, test loss_ce: %f, '
                        'test loss_dice: %f, test dice : %f' % 
                        (iter_num, loss.item(), loss_ce.item(), 
                         loss_dice.item(), dice.item())
                    )

                iter_num += 1

        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))
        
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]

        logger.info(
            'epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % 
            (cur_epoch, mean_loss, mean_loss_ce, mean_dice)
        )
        print(
            'Mean Test loss : %f, mean Test ce loss: %f, Mean Test dice : %f' % 
            (mean_loss, mean_loss_ce, mean_dice)
        )
        return class_dice, mean_dice
    
    def state_dict(self):
        state = {
            "model": self.model.state_dict(), 
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        return state
        
    def load_body(self, model_dict):
        new_state = {}
        for k, v in model_dict.items():
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
            
        if self.opts.step==1:
            print("Model changed, stochastic classifier replaces last layer.")
            for name, param in path.items():
                if name.startswith('outc'):
                    continue
                if name in self.model.state_dict():
                    self.model.state_dict()[name].copy_(param)
                    self.teacher.state_dict()[name].copy_(param)
        else:
            for name, param in path.items():
                if name.startswith('outc'):
                    print(self.model.state_dict()[name][0:nc].shape, param.shape)
                    self.model.state_dict()[name][0:nc].copy_(param)
                    self.teacher.state_dict()[name][0:nc].copy_(param)
                    continue
                if name in self.model.state_dict():
                    self.model.state_dict()[name].copy_(param)
                    self.teacher.state_dict()[name].copy_(param)
        
        for name, param in self.model.named_parameters():
            if name.startswith('outc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    def load_dict(self, path, strict=True):
        body = self.load_body(path)
        self.model.load_state_dict(body, strict)
    
    def load_dict_full_model(self, path, strict=True):
        self.model.load_state_dict(path, strict)
        
    
    def save_protos(self, dataset, step):
        protos={}
        if step>0:
            if step==1:
                with open('saved_proto_step0.pkl', 'rb') as f:
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
            wc, bg = get_prototype(model, ds, c, self.device, 
                                  interpolate_label=False, return_all=False, background=True)
            print("#########Good here")
            wc=wc.view(-1, 1, 1, 1)
            bg=bg.view(-1, 1, 1, 1)
            print(wc.shape, bg.shape)
            if wc is not None:
                protos[c]=wc
                bgkey='bg'+str(c)
                protos[bgkey]=bg
            else:
                raise Exception(f"Unable to imprint weight of class {c} after {count} trials.")
            print("Prototype for class {} saved".format(c))
            sys.stdout.flush()
        for key, val in protos.items():
            print(key, val.shape)
        
        if step==0:
            with open('saved_proto_step0.pkl', 'wb') as f:
                pickle.dump(protos, f)
        else:
            with open('saved_proto.pkl', 'wb') as f:
                pickle.dump(protos, f)