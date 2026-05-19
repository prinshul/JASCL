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
import pickle
from networks.custom.signet import SignetConv3d
from copy import deepcopy
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
        self.sparsity=0.03
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
        if self.need_model_old:
            self.model_old = self.make_model(config_vit, is_old=not model_old_as_new)
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.to(device)
            self.model_old.eval()

        # xxx Set up optimizer
        self.train_only_novel = self.opts.train_only_novel
        

        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opts.lr, momentum=0.9, weight_decay=self.opts.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 25, eta_min=5e-6)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)
        
        self.base_mask = torch.load('/hdd2/cil/baselines_by_Kaustubh/Softnet/inc_few_shot_medical/vanilla_organ/checkpoints/step/15-5-merged/Experiment_0_newUNET_dynamic.pth')
            
    def make_model(self, config_vit, is_old=False):
        # classifier, self.n_channels = self.get_classifier(is_old)
        n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
        model = make_model(self.opts, config_vit, n_classes[0])
        return model


    def percentile(self, scores, sparsity):
        k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
        return scores.view(-1).kthvalue(k).values.item()

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
        
    def train(self, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path = None):
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
        #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        # model.train()

        # for iteration in range(n_iter):
            # train_loader.sampler.set_epoch(cur_epoch*n_iter + iteration)  # setup dataloader sampler
        for i_batch, sampled_batch in enumerate(train_loader):
            
            # print("Patient ids", sampled_batch['case_name'])
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("Ground Truth", torch.unique(label_batch))

            #rloss = torch.tensor([0.]).to(self.device)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if self.opts.step==0:
                    #print(image_batch.shape, outputs.shape)
                    #print("Prediction", torch.unique(outputs))
                    # outputs, feat = model(image_batch)
                    emb, outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)
                    loss_dice = loss_dice_1

                    loss = 0.5*loss_ce+0.5*loss_dice
                    loss_tot = loss
                    dice = 1 - loss_dice_1
        
                    mean_loss += loss
                    mean_dice += dice
                    mean_loss_ce += loss_ce
                            
                    
                    optimizer.zero_grad()
                    loss_tot.backward()
                    optimizer.step()
            else:
                # optimize_incremental(self, optimizer, image_batch, label_batch)
                # set zero grads
                optimizer.zero_grad()

                #model = getattr(self.net_g, 'func')
                model.train()
                
                per_task_mask = model.get_masks_util(1)
                    
                per_new_task, self.grad_mask, self.old_mask  = self.get_soft_masks(mask=per_task_mask,
                                                                    sparsity=1-self.sparsity)
                
                
                # SubNet w/o per_task_mask for feature diversity
                 #self.base_mask[0] if True else None
                with torch.no_grad():
                    for name, module in self.model.named_modules():
                        if isinstance(module, SignetConv3d):
                            #print(name)
                            module.weight_mask = per_task_mask[name]
                            if getattr(module, 'bias') is not None:
                                #print("Bias Yes")
                                module.bias_mask = per_task_mask[name + 'bias']
                 
                

                l_total = 0
                # if self.cpn_loss_func is not None :
                # outputs, feat = model(image_batch)
                emb, outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)
                
                loss_dice = loss_dice_1
                    
                loss = 0.5*loss_ce+0.5*loss_dice
                loss_tot = loss
                dice = 1 - loss_dice_1
    
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce
                        

                l_total += loss
                #exit()

                
            
                l_total.backward(retain_graph=False)
                key_list = per_task_mask.keys()
                
                for key, param in model.named_parameters():
                    #print(key)
                
                    if ('down4' not in key) or ('norm' in key):
                        continue
                    
                    if not param.requires_grad:
                        continue

                    if 'w_m' in key :
                        param.grad = None

                    # if key in key_list:

                    m = self.grad_mask[key[:22]].detach()
                    mv = self.old_mask[key[:22]].detach()
                    # only target layer should be updated
                    if (key.startswith('down4')) and key.endswith('weight'):
                        #print(key)
                        if True:
                            # soft-scaled gradient
                            #print("Soft Scaled Grad")
                            # print(m*mv)
                            # exit()
                            param.grad *= m * mv
                        else:
                            # hard-scaled gradient
                            param.grad *= m
                    else:
                        param.grad = None

                optimizer.step()
            # update only minor params

            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)

            # loss_ce = loss_ce_1
            # loss_dice = loss_dice_1

            # if not self.opts.vanila:
            #     if self.model_old is not None:
            #         outputs_old, feat_old = self.model_old(image_batch)
            #         if self.kd_criterion is not None:
            #             kd_loss = self.kd_loss * self.kd_criterion(outputs, outputs_old)
            #             rloss += kd_loss
            #         if self.feat_criterion is not None:
            #             feat_loss = self.feat_loss * self.feat_criterion(feat, feat_old)
            #             rloss += feat_loss
            #         if self.de_criterion is not None:
            #             de_loss = self.de_loss * self.de_criterion(feat, feat_old)
            #             rloss += de_loss

                # print("KD loss = {}, Feature loss= {}, DE loss = {}".format(kd_loss, feat_loss, de_loss))
                # print("KD loss = {}".format(rloss))
            # loss = loss_ce
            
            # if not self.opts.vanila:
            #     loss_tot = loss + rloss
            #     if rloss <= CLIP:
            #         loss_tot = loss + rloss
            #     else:
            #         print(f"Warning, rloss is {rloss}! Term ignored")
            #         loss_tot = loss
                    
            #     loss_tot = loss + rloss
            # else:
            #     loss_tot = loss

            # loss_tot = loss
            # dice = 1 - loss_dice_1
            # mean_loss += loss
            # mean_dice += dice
            # mean_loss_ce += loss_ce
            
            #Training for base class
                
            # else:       ##Incremental steps
            #     incremental_optimize_parameters(self, optimizer)

            # _, prediction = outputs.max(dim=1)
            # print("Ground Truth", torch.unique(label_batch))
            # print("Prediction", torch.unique(prediction))
            

            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                    cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
                # No need for the below print statement, it is printing the same as the logger prints
                # print('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                #     cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice) 
                
            # saving per 100 iter nums
            if iter_num % 1000 == 0:
                    self.visualize_pred(label_batch, outputs, cur_epoch, iter_num)           
                
            # if(iter_num % 2 == 0):
            #     res_array = np.zeros(26).tolist()
            #     for i in range(26):
            #         if class_freq[i] != 0:
            #             res_array[i] = class_dice[i]/class_freq[i]
                        
        # mean loss and mean dice
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
        # mean_class_dice = list(map(lambda x: x/len(train_loader) if x != 'X' else x, class_dice))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                # print(class_freq[i])
                # print(class_dice[i])
                class_dice[i] = class_dice[i]/class_freq[i]
        
        # print('epoch :', cur_epoch, 'Train Loss :', mean_loss, 'Train dice :', mean_dice)
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
        #self.load_masks_into_model()
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

                
                iter_num += 1

                if iter_num % 3 == 0:
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
        self.load_masks_into_model()
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

                self.visualize_pred_test(label_batch, image_batch, outputs, i_batch)           

                
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
    
    def load_masks_into_model(self):
        fname='mask'+str(0)+'.pkl'
        print("Mask Loaded: ", fname)
        with open(fname, 'rb') as f:
            mask = pickle.load(f)
        
        for name, module in self.model.named_modules():
            if isinstance(module, SignetConv3d):
                #print(name)
                module.weight_mask = mask[name]
                if getattr(module, 'bias') is not None:
                    print("Bias Yes")
                    module.bias_mask = mask[name + '.bias']


    def load_dict_imprint(self, path, strict=True):

        if self.opts.step==1:
            nc=16
        elif self.opts.step==2:
            nc=21

        for name, param in path.items():
            if name.startswith('outc'):
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
        
        
        # for param in self.model.parameters():
        #     param.requires_grad = True
        fname='mask'+str(0)+'.pkl'
        print("Mask Loaded: ", fname)
        with open(fname, 'rb') as f:
            mask = pickle.load(f)
        
        for name, module in self.model.named_modules():
            if isinstance(module, SignetConv3d):
                print(name)
                module.weight_mask = mask[name]
                #print(mask[name])
                if getattr(module, 'bias') is not None:
                    print("Bias Yes")
                    module.bias_mask = mask[name + '.bias']      
            
        for name, param in self.model.named_parameters():
            if name.startswith('down4'):
                if name.endswith('w_m'):
                    param.requires_grad = False
                else:     
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
        self.load_masks_into_model()
        
    def optimize_parameters(self, current_iter, mask=None, smooth=False):
        self.optimizer_g.zero_grad()

        # ========== base learning ==========
        
        if smooth:
            self.net_g.func.apply(lambda m: setattr(m, "smooth", True))
        else :
            self.net_g.func.apply(lambda m: setattr(m, "smooth", False))

        self.score, original_output = self.net_g.forward_score(self.images,
                                                               mask = None,
                                                               mode = 'train')

        self.log_dict = AvgDict()
        loss = self.loss_func(original_output, self.labels_softmax)

        log_dict = {'CELoss': loss.item()}
        self.log_dict.add_dict(log_dict)
        loss.backward()
        self.optimizer_g.step()

        self.log_dict = self.log_dict.get_ordinary_dict()
    

    def get_curr_masks(self, sess_id):
        per_task_mask = deepcopy(self.model.get_masks(sess_id))
        return per_task_mask

    def get_masks(self, sess_id, layer, sparsity=0.3):
        if sess_id > 0:
            per_new_task, self.grad_mask, self.old_mask = self.get_soft_mask(
                mask=self.base_mask[0], sparsity=sparsity, layer=layer)

        elif sess_id == -1:
            per_new_task = self.get_curr_masks(0)
        else:
            per_new_task = self.base_mask

        return per_new_task
    
    def save_masks(self):
        return self.model.get_masks_util(0)


    def get_soft_masks(self, mask, sparsity=0.3,  scale=1.0):
        '''
        model         : base + shared + new
        mask          : base + shared
        new_task_mask : new  + shared
        base_mask     : mask - shared1
        new           : mask - shared
        '''
        model = self.model
        new_task_mask = deepcopy(mask)
        new_mask = deepcopy(mask)
        old_mask = deepcopy(mask)

        # avoid overlapped weights
        sparsity -= 0.0001  # for numerical stability
        new_sparsity = 1 # 100 % re-used ratio
        old_sparsity = 0
        for key in mask.keys():

            if 'bias' in key:
                continue

            if ('down4' not in key):
                continue

            new_mask[key] = torch.zeros_like(mask[key])
            #old_mask[key] = torch.zeros_like(mask[key])

            key_split = key.split('.')

            module_attr = 'w_m'

            # layer conv
            if len(key_split) == 4:
                curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])

            # up4 block
            elif len(key_split) == 5:
                #print(key_split)
                curr_module = getattr(getattr(getattr(model, key_split[0]), key_split[1]), key_split[2])
                curr_module=getattr(getattr(curr_module, key_split[3]), key_split[4])

            if hasattr(curr_module, module_attr):
                if getattr(curr_module, module_attr) is not None:
                    w_m = getattr(curr_module, module_attr)
                    #print("Shape of Score Matrix")
                    #print(w_m.shape)
                    # acquire one-hot mask
                    ones = torch.ones_like(w_m)
                    zeros = torch.zeros_like(w_m)

                    k_val = self.percentile(mask[key], sparsity * 100)
                    h_mask = torch.where(mask[key] < k_val, ones, zeros)

                    # get new weights
                    ones = torch.ones_like(w_m[h_mask==1])
                    zeros = torch.zeros_like(w_m[h_mask==1])

                    # update per_task_mask
                    new_task_m = torch.zeros_like(w_m)

                    k_val = self.percentile(mask[key][h_mask==1], new_sparsity * 100)
                    new_m = torch.where(mask[key][h_mask==1] < k_val, ones, zeros)

                    # update per_task_mask
                    new_task_m[h_mask==0] = mask[key][h_mask==0]
                    new_task_m[h_mask==1] = mask[key][h_mask==1]

                    # get new weights
                    ones = torch.ones_like(w_m[h_mask==0])
                    zeros = torch.zeros_like(w_m[h_mask==0])

                    k_val = self.percentile(mask[key][h_mask==0], old_sparsity * 100)
                    old_m = torch.where(mask[key][h_mask==0] < k_val, ones, zeros)

                    new_task_mask[key] = new_task_m
                    new_mask[key][h_mask==1] = new_m
                    old_mask[key][h_mask==0] = old_m

        #print("Success Getting Soft Masks")
        
        return new_task_mask, new_mask, old_mask
    #def optimize_incremental(self, optimizer, x, label):
        
