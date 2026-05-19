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
CLIP = 10

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class Trainer:
    def __init__(self, task, device, logger, opts, config_vit, class_wise_embeddings_previous):
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
        self.model, self.model_etf = self.make_model(config_vit, class_wise_embeddings_previous)
        self.model = self.model.to(device)
        self.model_etf = self.model_etf.to(device)
        
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
            self.model_old = self.make_model(config_vit, class_wise_embeddings_previous,is_old=not model_old_as_new)
            # put the old model into distributed memory and freeze it
            for par in self.model_old[0].parameters():
                par.requires_grad = False
            self.model_old[0].to(device)
            self.model_old[0].eval()

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
            
    def make_model(self, config_vit, class_wise_embeddings_previous, is_old=False):
        # classifier, self.n_channels = self.get_classifier(is_old)
        n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
        model = make_model(self.opts, config_vit, n_classes[0], class_wise_embeddings_previous)
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
        
        sitk.WriteImage(label_arr, os.path.join('cil/FSS_26_TS/data/merged/dataset/prediction/ground_truth','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('cil/FSS_26_TS/data/merged/dataset/prediction/predicted_mask','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))

    def visualize_pred_test(self, labels, images, prediction, i_batch):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        image_arr = sitk.GetImageFromArray(images.detach().cpu().numpy())

        sitk.WriteImage(label_arr, os.path.join('cil/FSS_26_TS/data/merged/dataset/prediction/test/ground_truth','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('cil/FSS_26_TS/data/merged/dataset/prediction/test/predicted_mask','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(image_arr, os.path.join('cil/FSS_26_TS/data/merged/dataset/prediction/test/images','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        
    def train(self, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path = None):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = optimizer
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        
        model = self.model
        model_etf = self.model_etf
        
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        # for name, param in model_etf.named_parameters():
        #     print(name, param.requires_grad)
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

        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            torch.cuda.empty_cache()
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            patch_embeddings, outputs = model(image_batch)
            class_mean_calculator, loss_etf1, loss_etf2 = model_etf(image_batch, label_batch, patch_embeddings)
           
            loss_ce_1 = ce_loss(outputs, label_batch[:].long())
            class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)

            loss_ce = loss_ce_1
            loss_dice = loss_dice_1
            if loss_etf2 is None:
                loss = loss_ce + loss_dice + 10*loss_etf1
            else:
                loss = loss_ce + loss_dice + 10*loss_etf1 + 10*loss_etf2
            dice = 1 - loss_dice_1
            mean_loss += loss
            mean_dice += dice
            mean_loss_ce += loss_ce

            loss.backward()
           
            # Perform gradient update
            optimizer.step()
            optimizer.zero_grad()
            
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                    cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))

            if metrics is not None:
                class_dice = metrics.calculate_dice_each_class(class_wise_dice) 
            torch.cuda.empty_cache()
            
        # mean loss and mean dice
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
        # mean_class_dice = list(map(lambda x: x/len(train_loader) if x != 'X' else x, class_dice))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
        
        # print('epoch :', cur_epoch, 'Train Loss :', mean_loss, 'Train dice :', mean_dice)
        logger.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))

        save_interval = 50  # int(max_epoch/5)
        # if cur_epoch > int(max_epoch / 5) and (cur_epoch + 1) % save_interval == 0:
        # if (cur_epoch + 1) >= 350 and (cur_epoch + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(cur_epoch + 1) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logger.info("save model to {}".format(save_mode_path))

        # if cur_epoch + 1 == max_epoch:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(cur_epoch + 1) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logger.info("save model to {}".format(save_mode_path))

        return class_dice, mean_dice, class_mean_calculator, model_etf

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
                _, outputs = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch, self.total_classes, softmax=True)


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
                    logger.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
                    print('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))

         
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
                _, outputs = model(image_batch)
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

                # self.visualize_pred_test(label_batch, image_batch, outputs, i_batch)           

                
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
    

    def load_dict_imprint(self, path, class_mean_calculator, strict=True):
        for name, param in path.items():
            if name.startswith('outc'):
                continue  # Skip the final layer weights
            if name in self.model.state_dict():
                self.model.state_dict()[name].copy_(param)
        
        # tmp = class_mean_calculator[:self.opts.num_classes].data.clone()
        # tmp = tmp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # self.model.outc.weight.data = tmp
        
        for name, param in self.model.named_parameters():
            if name.startswith('up') or name.startswith('outc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        
    def load_dict(self, path, strict=True):
        body = self.load_body(path)
        self.model.load_state_dict(body, strict)
    
    def load_dict_full_model(self, path, strict=True):
        self.model.load_state_dict(path, strict)
        
