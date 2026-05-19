import torch
from torch import distributed
import torch.nn.functional as F

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from utils.loss import KnowledgeDistillationLoss, CosineLoss, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, CosineKnowledgeDistillationLoss, Multi_BCELoss, DiceLoss1
from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier, CosineClassifier, SPNetClassifier
from .utils import get_scheduler, MeanReduction
from networks.VerSe.unet import network as network
from torch.nn.modules.loss import CrossEntropyLoss
from utils.VerSe_utils import DiceLoss, print_network
import SimpleITK as sitk
import numpy as np
import os
import random
from monai.inferers import sliding_window_inference
from methods.segmentation_module import get_any_model
CLIP = 10

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
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

        self.n_channels = -1  # features size, will be initialized in make model
        self.model = self.make_model(config_vit, is_old = False)
        self.model = self.model.to(device)

        self.distributed = False
        self.model_old = None

        if self.opts.fix_bn:
            self.model.fix_bn()

        if self.opts.bn_momentum is not None:
            self.model.bn_set_momentum(self.opts.bn_momentum)


        self.born_again = self.opts.born_again
        self.dist_warm_start = self.opts.dist_warm_start
        model_old_as_new = self.opts.born_again or self.opts.dist_warm_start
        if self.need_model_old:
            if self.task.step > 0:
                task_name = f"{opts.task}-{opts.dataset}"
                name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"

                if opts.step - 1 == 0:
                    checkpoint_path_old = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}_{opts.network_arch}_dynamic.pth"
                else:
                    checkpoint_path_old = f"checkpoints/step/{task_name}/{name}_{opts.step - 1}_{opts.network_arch}_dynamic.pth"
            
            # checkpoint_path_old = torch.load('checkpoints/step/15ss-merged/FT_0_SwinUNETR_partial_dynamic.pth', map_location="cpu")

            self.model_old = self.make_model(config_vit, is_old=True)
            self.load_dict(checkpoint_path_old)
            logger.info(f"*** Old Model restored")

            
            if self.opts.network_arch == 'SwinUNETR_partial' and self.opts.trans_encoding == 'word_embedding':
                word_embedding = torch.load(self.opts.word_embedding)
                self.model.organ_embedding.data = word_embedding.float()
                self.model_old.organ_embedding.data = word_embedding.float()

                print('..........load word embedding..............')

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
        n_classes = self.opts.num_classes - self.novel_classes if is_old else self.opts.num_classes
        model = make_model(self.opts, config_vit, n_classes)
        return model

    def distribute(self):
        self.opts = self.opts
        if self.model is not None:
            # Put the model on GPU
            self.distributed = True
            self.model = DistributedDataParallel(self.model, device_ids=[self.opts.device_id],
                                                 output_device=self.opts.device_id, find_unused_parameters=True)


    def warm_up(self, dataset, epochs=1):
        self.warm_up_(dataset, epochs)
        # warm start means make KD after weight imprinting or similar
        if self.dist_warm_start:
            self.model_old.load_state_dict(self.model.state_dict())

    def warm_up_(self, dataset, epochs=1):
        pass

    def cool_down(self, dataset, epochs=1):
        pass
    
    def visualize_pred(self, label_batch, labels, prediction, cur_epoch, cur_step):
        if prediction is not None:
            prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        original_arr = sitk.GetImageFromArray(label_batch.detach().cpu().numpy())

        if prediction is not None:
            sitk.WriteImage(prediction_arr, os.path.join('data/merged/prediction/predicted_mask','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))

        sitk.WriteImage(label_arr, os.path.join('data/merged/prediction/ground_truth','{}_{}_{}_new.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))
        sitk.WriteImage(original_arr, os.path.join('data/merged/prediction/original','{}_{}_{}_new.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))

    def visualize_pred_test(self, labels, images, prediction, i_batch):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        image_arr = sitk.GetImageFromArray(images.detach().cpu().numpy())

        sitk.WriteImage(label_arr, os.path.join('cil/JHH_new/organ/data/merged/prediction/test/ground_truth','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('cil/JHH_new/organ/data/merged/prediction/test/predicted_mask','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(image_arr, os.path.join('cil/JHH_new/organ/data/merged/prediction/test/images','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
    def load_vanilla_base_model(self):
        self.vanilla_base_model = get_any_model(network_arch = "newUNET", num_classes = 16)
        checkpoint_path_old = 'cil/IFSS/organ/checkpoints/step/15ss-merged/FT_0_newUNET_dynamic.pth'
        ckpt = torch.load(checkpoint_path_old, map_location="cpu")
        self.vanilla_base_model.load_state_dict(ckpt['model_state']['model'], strict =  True)
        
        print("*** Old Vanilla Base Model restored")

    def train(self, model, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path = None):
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
        
        self.load_vanilla_base_model()
        
        model.train()
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        dice_loss_2 = DiceLoss1()
        multi_bce = Multi_BCELoss()

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        loss_bce_ave = 0
        loss_dice_ave = 0
    
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            label_unique = torch.unique(label_batch)
            # novel_labels_set = set([x for x in label_unique if x > len(self.labels_old)])
            # old_labels_set = set([x for x in label_unique if x <= len(self.labels_old)])
            outputs =  self.model(image_batch)[-1]

            if self.model_old is not None:
                output_old = self.model_old(image_batch)
                
                output_old_softmax = torch.softmax(output_old, dim=1)
                output_mask_old = torch.argmax(output_old_softmax[:, 1:], dim=1)
                output_mask_old = output_mask_old.unsqueeze(1).float()
                
                condition = (label_batch <= len(self.labels_old)) & (label_batch > 0)
                # print("Before:", torch.unique(label_batch))
                label_batch_new = torch.where(condition, output_mask_old, label_batch)
                # print("After SwinUNETR pseudo labels:", torch.unique(label_batch_new))
                condition_new = (label_batch_new < len(self.labels_old)) & (label_batch_new > 0)

                label_batch_final = torch.where(condition_new, label_batch_new + 1, label_batch_new)
            else:
                label_batch_final = label_batch
            if self.opts.load_vanilla_base_model:
                outputs_vanilla_base, _ =  self.vanilla_base_model(image_batch)
                outputs_vanilla_base_softmax = torch.softmax(outputs_vanilla_base, dim=1)
                outputs_vanilla_base_mask = torch.argmax(outputs_vanilla_base_softmax[:, 1:], dim=1)
                outputs_vanilla_base_mask = outputs_vanilla_base_mask.unsqueeze(1).float()

                label_batch_vanilla_new = torch.where(condition, outputs_vanilla_base_mask, label_batch)
                # print("After 3DUnet pseudo labels:", torch.unique(label_batch_vanilla_new))
                

                if iter_num % 400 == 0:
                    self.visualize_pred(label_batch, label_batch_final, label_batch_vanilla_new, cur_epoch, iter_num)
          
            if iter_num % 400 == 0:
                self.visualize_pred(label_batch, label_batch_final, None, cur_epoch, iter_num)

            # print(2/0)
            if self.opts.out_nonlinear == 'sigmoid':
                # print(torch.unique(outputs), torch.unique(label_batch))
                # image_batch, label_batch = image_batch.squeeze(0), label_batch.squeeze(0)
                target_one_hot = F.one_hot(label_batch_final.long(), num_classes=num_classes)
                target_one_hot = target_one_hot.permute(0, 1, 5, 2, 3, 4)
                term_seg_BCE = multi_bce(outputs, target_one_hot, num_classes)

                # term_seg_Dice = dice_loss_2(outputs, target_one_hot, num_classes)
                # loss_dice_ave += term_seg_Dice.item()
           
            
            loss_ce_1 = ce_loss(outputs, label_batch_final[:].long().squeeze(1))
            class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch_final[:].long(), self.total_classes, softmax=True)

            # loss_ce = loss_ce_1
            loss_dice = loss_dice_1
            # loss = 0.5 * loss_dice + 0.5 * term_seg_BCE
            loss = 0.5*loss_dice_1 + 0.4*loss_ce_1  + 0.1*term_seg_BCE
            # dice_2 = 1 - term_seg_Dice

            dice = 1 - loss_dice_1
            mean_loss += loss
            mean_dice += dice

            # mean_loss_ce += loss_ce
            mean_loss_ce += term_seg_BCE
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
                        
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_dice: %f, train BCE loss : %f, train dice : %f' % (
                    cur_epoch, iter_num, loss.item(), loss_dice_1.item(), term_seg_BCE.item(), dice.item()))
                # logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_dice: %f, train BCE loss : %f, train dice : %f' % (
                #     cur_epoch, iter_num, loss.item(), loss_dice.item(), term_seg_BCE.item(), dice.item()))
                # No need for the below print statement, it is printing the same as the logger prints
                # print('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                #     cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice) 
                
            # saving per 100 iter nums
            # if iter_num % 1000 == 0:
            #         self.visualize_pred(label_batch, outputs, cur_epoch, iter_num)           
    
            torch.cuda.empty_cache()
        # mean loss and mean dice
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
        
        logger.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        # print('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
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
        multi_bce = Multi_BCELoss()
        dice_loss_2 = DiceLoss1()

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                # outputs, _ = model(image_batch)
                # outputs =  model(image_batch)[-1]
                pred = sliding_window_inference(image_batch, (96, 96, 96), 1, model, overlap=0.5, mode='gaussian')

                if self.opts.out_nonlinear == 'sigmoid':
                    pred_sigmoid = F.sigmoid(pred)
                    #pred_hard = threshold_organ(pred_sigmoid, organ=args.threshold_organ, threshold=args.threshold)
                    # pred_hard = threshold_organ(pred_sigmoid)
                    pred_hard = pred_sigmoid > 0.5
                # print(pred_hard.shape)
                # if self.opts.out_nonlinear == 'sigmoid':
                #     target_one_hot = F.one_hot(label_batch, num_classes=num_classes)
                #     target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3)
                #     term_seg_Dice = dice_loss_2(pred_hard, target_one_hot, num_classes)
               

                #     term_seg_BCE = multi_bce(pred_hard, target_one_hot, num_classes)
                #     loss_bce_ave += term_seg_BCE.item()
                # loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(pred, label_batch, self.total_classes, softmax=True)


                # loss_ce = loss_ce_1
            
                loss_dice = loss_dice_1
                # dice_2 = 1 - term_seg_Dice
                dice = 1 - loss_dice_1
                # loss = 0.5 * loss_dice + 0.5 * term_seg_BCE
                # mean_loss += loss
                mean_dice += dice
                # mean_loss_ce += loss_ce
                # mean_loss_ce = term_seg_BCE
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                
                iter_num += 1

                if iter_num % 3 == 0:
                    logger.info('epoch : %d, iteration : %d, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss_dice.item(), dice.item()))
                    # print('epoch : %d, iteration : %d, val loss : %f, val loss_dice: %f, val dice : %f, val BCE loss : %f' % (
                    #     cur_epoch, iter_num, loss.item(), loss_dice.item(), dice.item(), term_seg_BCE.item()))

    
        # mean loss and mean dice
        # mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        # mean_loss_ce = float(mean_loss_ce / len(loader))
        # mean_class_dice = list(map(lambda x: x/len(loader) if x != 'X' else x, class_dice))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
                
        logger.info('epoch : %d, mean val dice : %f' % (cur_epoch, mean_dice))
        # print('epoch : %d, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def test(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, novel=False, cur_epoch=None, snapshot_path = None, test_old = False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        # criterion = self.criterion
        logger = self.logger

        model.eval()
        iter_num = cur_epoch * len(loader)
        if test_old:
            model = self.model_old
            num_classes = len(self.labels_old) + 1 
            total_classes = self.labels_old
        else:
            num_classes = self.opts.num_classes
            total_classes = self.total_classes
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        dice_loss_2 = DiceLoss1()

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                # outputs, _ = model(image_batch)
                # outputs =  model(image_batch)[-1]
                pred = sliding_window_inference(image_batch, (96, 96, 96), 1, model, overlap=0.5, mode='gaussian')

                if self.opts.out_nonlinear == 'sigmoid':
                    pred_sigmoid = torch.sigmoid(pred)
                    #pred_hard = threshold_organ(pred_sigmoid, organ=args.threshold_organ, threshold=args.threshold)
                    # pred_hard = threshold_organ(pred_sigmoid)
                    pred_hard = pred_sigmoid > 0.5
                # if self.opts.out_nonlinear == 'sigmoid':
                #     target_one_hot = F.one_hot(label_batch, num_classes=num_classes)
                #     target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3)
                #     term_seg_Dice = dice_loss_2(pred_hard, target_one_hot, num_classes)
                #     term_seg_BCE = multi_bce(pred_hard, target_one_hot, num_classes)
                #     loss_bce_ave += term_seg_BCE.item()
                # print("Ground Truth", torch.unique(label_batch))
                # loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(pred, label_batch, total_classes, softmax=True)

                # loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                # dice_2 = 1 - term_seg_Dice
                # loss = 0.5 * loss_ce + 0.5 * loss_dice
                # loss = 0.5 * loss_dice + 0.5 * term_seg_BCE
                # mean_loss += loss
                mean_dice += dice
                # mean_loss_ce += loss_ce
                # mean_loss_ce = term_seg_BCE

                    
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                # self.visualize_pred_test(label_batch, image_batch, outputs, i_batch)           

                
                if iter_num % 5 == 0:
                    logger.info('iteration : %d, test loss_dice: %f, test dice : %f ' % (
                        iter_num, loss_dice.item(), dice.item()))

                    # print('iteration : %d, test loss : %f, test loss_dice: %f, test dice : %f, test BCE loss : %f' % (
                    #     iter_num, loss.item(), loss_dice.item(), dice.item(), term_seg_BCE.item()))


                iter_num += 1

        # mean loss and mean dice
        # mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        # mean_loss_ce = float(mean_loss_ce / len(loader))
        
        # print(class_dice)
        # print(class_freq)
        for i in range(num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]

        # mean_class_dice = list(map(lambda x: x/len(loader) if x != 'X' else x, class_dice))

        logger.info('epoch : %d,  mean test dice : %f' % (cur_epoch, mean_dice))
        # print('Mean Test loss : %f, mean Test ce loss: %f, Mean Test dice : %f' % (mean_loss, mean_loss_ce, mean_dice))
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
        for name, param in path.items():
            if name.startswith('outc'):
                continue  # Skip the final layer weights
            if name in self.model.state_dict():
                self.model.state_dict()[name].copy_(param)
        
        for name, param in self.model.named_parameters():
            if name.startswith('outc') or name.startswith('up'):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def load_dict(self, path, strict=True):
        step_checkpoint = torch.load(path, map_location="cpu")
        # self.model_old.load_state_dict(step_checkpoint['model_state']['model'], strict=True)  # Load also here old parameters
        for name, param in step_checkpoint['model_state']['model'].items():
            if not 'organ_embedding' in name:
                self.model_old.state_dict()[name].copy_(param)
       
    def load_dict_full_model(self, path, strict=True):
        for name, param in path.items():
            if not 'organ_embedding' in name:
               self.model.state_dict()[name].copy_(param)
        
        if self.opts.network_arch == 'SwinUNETR_partial' and self.opts.trans_encoding == 'word_embedding':
            word_embedding = torch.load(self.opts.word_embedding)
            self.model.organ_embedding.data = word_embedding.float()

            print('..........load word embedding..............')