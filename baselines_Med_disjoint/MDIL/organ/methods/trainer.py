import torch
from torch import distributed
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from utils.loss import KnowledgeDistillationLoss, CosineLoss, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, CosineKnowledgeDistillationLoss
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
from methods.segmentation_module import get_any_model
from torch.optim import SGD, Adam, lr_scheduler
import re

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
        self.model = self.make_model(config_vit)
        self.model = self.model.to(device)

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
        self.model_old = None
            
        # xxx Set up optimizer
        self.train_only_novel = self.opts.train_only_novel
        self.old_classes = len(self.labels_old)
        
        self.lkd_flag = False
        self.lkd = 0.1 
        
        if self.task.step > 0:
            task_name = f"{opts.task}-{opts.dataset}"
            name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"

            if opts.step - 1 == 0:
                checkpoint_path_old = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}_{opts.network_arch}_dynamic.pth"
            else:
                checkpoint_path_old = f"checkpoints/step/{task_name}/{name}_{opts.step - 1}_{opts.network_arch}_dynamic.pth"
                
            self.model_old = self.make_model(config_vit, is_old=True)
            step_ckpt_old = self.load_dict(checkpoint_path_old)
            logger.info(f"*** Old Model restored")
            
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
                
            self.model_old.to(device)
            self.model_old.eval()
            
            checkpoint_path_old_save_new = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}_{opts.network_arch}_dynamic_old_save_new.pth"
            self.save_ckpt(checkpoint_path_old_save_new, self.model_old, -1)

            for name, param in self.model.named_parameters():
                print(name)

            # print("**********************************")
                
            new_dict_load = self.init_new_model(step_ckpt_old, logger)
            self.model.load_state_dict(new_dict_load, strict=False)
            self.freeze_new_model()
            self.lkd_flag =  True
            
            for name, param in self.model.named_parameters():
                print(name, param.requires_grad)
                
        if opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=len(self.labels_old), ignore_index=255, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
               
        # self.optimizer = torch.optim.SGD(params, lr=self.opts.lr, momentum=0.9, weight_decay=self.opts.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 25, eta_min=5e-6)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)
       
        # Feature distillation
        self.kd_criterion = torch.nn.KLDivLoss()
        self.kd_criterion = self.kd_criterion.cuda()
        
            
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
    def save_ckpt(self, path, model, epoch):
        """ save current model
        """
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
        }

        torch.save(state, path)
        
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
        
        sitk.WriteImage(label_arr, os.path.join('cil/MDIL/organ/data/merged/prediction/ground_truth','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('cil/MDIL/organ/data/merged/prediction/predicted_mask','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))

    def visualize_pred_test(self, labels, images, prediction, i_batch):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        image_arr = sitk.GetImageFromArray(images.detach().cpu().numpy())

        sitk.WriteImage(label_arr, os.path.join('cil/MDIL/organ/data/merged/prediction/test/ground_truth','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('cil/MDIL/organ/data/merged/prediction/test/predicted_mask','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(image_arr, os.path.join('cil/MDIL/organ/data/merged/prediction/test/images','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
    
    def generate_mask(self, model, outputs_vanilla_base):
        outputs_vanilla_base_softmax = torch.softmax(outputs_vanilla_base, dim=1)
        
        return outputs_vanilla_base_softmax
    
    def train(self, optimizer, lr_scheduler, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path = None):
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

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0

        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        # print("**********************************")
            
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            rloss = torch.tensor([0.]).to(self.device)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            if self.lkd_flag and self.model_old is not None:
                with torch.no_grad():
                    outputs_old_prev_step = self.model_old(image_batch, self.opts.step - 1)
                    outputs_new_prev_step = self.model(image_batch, self.opts.step - 1)

            
            outputs = model(image_batch,  self.opts.step)

            loss = ce_loss(outputs, label_batch)
            
            if self.lkd_flag:
                outputs_mask_old_prev_step = self.generate_mask(self.model_old, outputs_old_prev_step)
                outputs_mask_new_prev_step = self.generate_mask(self.model_old, outputs_new_prev_step)
                
                KLD_loss = self.kd_criterion(outputs_mask_old_prev_step, outputs_mask_new_prev_step)
                # resize new output to remove new logits and keep only the old ones
                lkd = self.lkd * KLD_loss
            else:
                lkd  = 0
            class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch, self.total_classes, softmax=True)

            loss_dice = loss_dice_1
            loss_tot = 0.3*loss + 0.4*lkd + 0.3*loss_dice
            
            optim.zero_grad()
            loss_tot.backward()
            optim.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            dice = 1 - loss_dice_1
            mean_loss += loss_tot
            mean_dice += dice
            mean_loss_ce += loss
            
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
                
            if iter_num % 200 == 0:
                self.visualize_pred(label_batch, outputs, cur_epoch, iter_num)

            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                    cur_epoch, iter_num, loss.item(), loss.item(), loss_dice.item(), dice.item()))

            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice) 

        # mean loss and mean dice
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
        
        logger.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))

        save_interval = 50  # int(max_epoch/5)
        # if cur_epoch > int(max_epoch / 5) and (cur_epoch + 1) % save_interval == 0:
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
                
                if self.lkd_flag and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old_prev_step = self.model_old(image_batch, self.opts.step - 1)
                        outputs_new_prev_step = self.model(image_batch, self.opts.step - 1)

            
                outputs = model(image_batch,  self.opts.step)

                loss = ce_loss(outputs, label_batch)
            
                if self.lkd_flag:
                    outputs_mask_old_prev_step = self.generate_mask(self.model_old, outputs_old_prev_step)
                    outputs_mask_new_prev_step = self.generate_mask(self.model_old, outputs_new_prev_step)
                    
                    KLD_loss = self.kd_criterion(outputs_mask_old_prev_step, outputs_mask_new_prev_step)
                    # resize new output to remove new logits and keep only the old ones
                    lkd = self.lkd * KLD_loss
                else:
                    lkd  = 0
                    
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)
                loss_dice = loss_dice_1

                loss_tot = 0.3*loss + 0.4*lkd + 0.3*loss_dice

                dice = 1 - loss_dice_1
                mean_loss += loss_tot
                mean_dice += dice
                mean_loss_ce += loss

                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                
                iter_num += 1

                if iter_num % 3 == 0:
                    logger.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss.item(), loss_dice.item(), dice.item()))
                    print('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss.item(), loss_dice.item(), dice.item()))

    
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
                
                if self.lkd_flag and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old_prev_step = self.model_old(image_batch, self.opts.step - 1)
                        outputs_new_prev_step = self.model(image_batch, self.opts.step - 1)

            
                outputs = model(image_batch,  self.opts.step)

                loss = ce_loss(outputs, label_batch)
            
                if self.lkd_flag:
                    outputs_mask_old_prev_step = self.generate_mask(self.model_old, outputs_old_prev_step)
                    outputs_mask_new_prev_step = self.generate_mask(self.model_old, outputs_new_prev_step)
                    
                    KLD_loss = self.kd_criterion(outputs_mask_old_prev_step, outputs_mask_new_prev_step)
                    # resize new output to remove new logits and keep only the old ones
                    lkd = self.lkd * KLD_loss
                else:
                    lkd  = 0
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)
                loss_dice = loss_dice_1
                loss_tot = 0.3*loss + 0.4*lkd + 0.3*loss_dice

                dice = 1 - loss_dice_1
                mean_loss += loss_tot
                mean_dice += dice
                mean_loss_ce += loss

                    
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                # self.visualize_pred_test(label_batch, image_batch, outputs, i_batch)           

                
                if iter_num % 5 == 0:
                    print('iteration : %d, test loss : %f, test loss_ce: %f, test loss_dice: %f, test dice : %f' % (
                        iter_num, loss.item(), loss.item(), loss_dice.item(), dice.item()))


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
        
    def load_dict(self, path, strict=True):
        step_checkpoint = torch.load(path, map_location="cpu")
        # self.model_old.load_state_dict(step_checkpoint['model_state']['model'], strict)
        for name, param in step_checkpoint['model_state']['model'].items():
            # if not name.startswith('seg_head'):
            self.model_old.state_dict()[name].copy_(param)
        return step_checkpoint['model_state']['model']

    def load_dict_full_model(self, path, strict=True):
            self.model.load_state_dict(path, strict)
            
    def init_new_model(self, step_ckpt_old, logger):
        new_dict_load = {}
        for k, v in step_ckpt_old.items():
            if k in self.model.state_dict().keys():  # take all the common params as it is
                new_dict_load[k] = v

        logger.info('Copying the Step {}-RAPs into Step {}-RAPs as initialisation (to avoid random init)'.format(self.opts.step - 1, self.opts.step))
        # print('Not copying BN layers, they are randomly init.\n\n')
        logger.info('copying decoder but not output_conv of previous step {} into current step {}'.format(
            self.opts.step - 1, self.opts.step))

        # put all the previous task's DS params into current tasks DS params. being used as an init strategy
        for k, v in step_ckpt_old.items():
            if 'encoder' in k:
                if 'parallel_conv' in k or 'bn' in k:
                    if '.{}.weight'.format(self.opts.step-1) in k:
                        # print("...", k)

                        nkey = re.sub('.{}.weight'.format(self.opts.step-1),
                                        '.{}.weight'.format(self.opts.step), k)
                        new_dict_load[nkey] = v
                        # print(nkey)
                    elif '.{}.bias'.format(self.opts.step-1) in k:
                        nkey = re.sub('.{}.bias'.format(self.opts.step-1),
                                        '.{}.bias'.format(self.opts.step), k)
                        new_dict_load[nkey] = v

            elif 'decoder' in k and 'output_conv' not in k:
                # this is important so as to maintain uniformity among bdd and idd experiments.
                nkey = re.sub('decoder.{}'.format(self.opts.step-1),
                                'decoder.{}'.format(self.opts.step), k)
                new_dict_load[nkey] = v
            
            # elif 'seg_head' in k:
            #     nkey = re.sub('seg_head.{}'.format(self.opts.step-1),
            #                     'seg_head.{}'.format(self.opts.step), k)
            #     if 'weight' in k:
            #         if self.opts.step == 0:
            #             new_dict_load[nkey] = self.model.seg_head[0].weight
            #         elif self.opts.step == 1:
            #             new_dict_load[nkey] = self.model.seg_head[1].weight
            #         else:
            #             new_dict_load[nkey] = self.model.seg_head[2].weight
                    
            #         new_dict_load[nkey][:len(self.labels_old) + 1, :, :, : ,:].data = v
                
            #     if 'bias' in k:
            #         if self.opts.step == 0:
            #             new_dict_load[nkey] = self.model.seg_head[0].bias
            #         elif self.opts.step == 1:
            #             new_dict_load[nkey] = self.model.seg_head[1].bias
            #         else:
            #             new_dict_load[nkey] = self.model.seg_head[2].bias
                
            #         new_dict_load[nkey][:len(self.labels_old) + 1].data = v
            
        # new_dict_load['seg_head.weight'] = self.model.seg_head.weight
        # new_dict_load['seg_head.bias'] = self.model.seg_head.bias
        
        # new_dict_load['seg_head.weight'][:len(self.labels_old) + 1, :, :, : ,:].data = self.model_old.seg_head.weight.data
        # new_dict_load['seg_head.bias'][:len(self.labels_old) + 1].data = self.model_old.seg_head.bias.data
        
        return new_dict_load
    
    def freeze_new_model(self):
        
        if self.opts.network_arch == 'erfnet_RA_parallel':
            if self.opts.step == 0:
                for name, m in self.model.named_parameters():
                    if 'decoder' in name:
                        if 'decoder.{}'.format(self.opts.step) in name:
                            m.requires_grad = True
                        else:
                            m.requires_grad = False

                    elif 'encoder' in name:
                        if 'bn' in name or 'parallel_conv' in name:
                            if '.{}.weight'.format(self.opts.step) in name or '.{}.bias'.format(self.opts.step) in name:
                                m.requires_grad = True
                            else:
                                m.requires_grad = False
            else:
                for name, m in self.model.named_parameters():
                    if 'decoder' in name:
                        if 'decoder.{}'.format(self.opts.step) not in name:
                            m.requires_grad = False

                    elif 'encoder' in name:
                        if 'bn' in name or 'parallel_conv' in name:
                            if '.{}.weight'.format(self.opts.step) in name or '.{}.bias'.format(self.opts.step) in name:
                                continue
                            else:
                                m.requires_grad = False
                    
                    elif 'seg_head' in name:
                        if '.{}.weight'.format(self.opts.step) in name or '.{}.bias'.format(self.opts.step) in name:
                            m.requires_grad = True
                        else:
                            m.requires_grad = False