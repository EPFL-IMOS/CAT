#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import models
import datasets
from loss.focal_loss import FocalLoss
from utils.entropy_CDA import Entropy
from utils.entropy_CDA import calc_coeff
from utils.entropy_CDA import grl_hook
from utils.self_training import *
from utils.calibration import *
from datasets.sequence_aug import *

from transcal.generate_features import generate_feature_wrapper

from loss.mcc import MinimumClassConfusionLoss
from optim.sam import SAM


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir=save_dir)
        self.alpha = args.alpha


    def update_teacher(self, alpha):
        """Update teacher with student parameters using EMA."""
        for teacher_param, param in zip(self.model_teacher_all.parameters(), self.model_all.parameters()):
            teacher_param.data[:] = alpha * teacher_param[:].data[:] + (1 - alpha) * param[:].data[:]

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}

        if isinstance(args.transfer_task[0], str):
            #print(args.transfer_task)
            args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)
        print([len(self.datasets[x]) for x in ['source_train', 'source_val', 'target_train', 'target_val']])
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Define the teacher and student models
        self.model = getattr(models, args.model_name)(pretrained=args.pretrained)
        self.model_teacher = getattr(models, args.model_name)(pretrained=args.pretrained)

        if args.bottleneck:
            if args.model_name in ["resnet101", "resnet50", "resnet18"]:
                self.bottleneck_layer = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.model.out_features, args.bottleneck_num),
                    nn.BatchNorm1d(args.bottleneck_num),
                    nn.ReLU(inplace=True)
                )
                self.bottleneck_layer_teacher = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.model_teacher.out_features, args.bottleneck_num),
                    nn.BatchNorm1d(args.bottleneck_num),
                    nn.ReLU(inplace=True)
                )
            else:
                self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                      nn.ReLU(inplace=True), nn.Dropout())
                self.bottleneck_layer_teacher = nn.Sequential(nn.Linear(self.model_teacher.output_num(), args.bottleneck_num),
                                                    nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
            self.classifier_layer_teacher = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)
            self.classifier_layer_teacher = nn.Linear(self.model_teacher.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        self.model_teacher_all = nn.Sequential(self.model_teacher, self.bottleneck_layer_teacher, self.classifier_layer_teacher)

        self.update_teacher(alpha=0.0)  # copy student model into teacher

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders['source_train'])*(args.max_epoch-args.middle_epoch)
            if args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                if args.bottleneck:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num*Dataset.num_classes,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
                else:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num()*Dataset.num_classes,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
            else:
                if args.bottleneck_num:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
                else:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )

        # if self.device_count > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        #     if args.bottleneck:
        #         self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
        #     if args.domain_adversarial:
        #         self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
        #     self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the learning parameters
        if args.bottleneck:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr if not args.pretrained else 0.1*args.lr},
                              {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        else:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr if not args.pretrained else 0.1*args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        if args.domain_adversarial:
            if args.sdat:
                parameter_list_ad = [{"params": self.AdversarialNet.parameters(), "lr": args.lr}]
                if args.opt == 'sgd':
                    self.optimizer_ad = torch.optim.SGD(parameter_list_ad, lr=args.lr,
                                            momentum=args.momentum, weight_decay=args.weight_decay)
                elif args.opt == 'adam':
                    self.optimizer_ad = torch.optim.Adam(parameter_list_ad, lr=args.lr,
                                                weight_decay=args.weight_decay)
                else:
                    raise Exception("optimizer not implement")
            else:
                parameter_list += [{"params": self.AdversarialNet.parameters(), "lr": args.lr}]

        # Define the optimizer
        if args.sdat:
            if args.opt == 'sgd':
                self.optimizer_sam = SAM(parameter_list, torch.optim.SGD, rho=0.002 if args.model_name.startswith("vit") else 0.05, lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.opt == 'adam':
                self.optimizer_sam = SAM(parameter_list, torch.optim.Adam, rho=0.002 if args.model_name.startswith("vit") else 0.05, lr=args.lr,
                    weight_decay=args.weight_decay)
            else:
                raise Exception("optimizer not implement")

        if args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(parameter_list, lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = torch.optim.Adam(parameter_list, lr=args.lr,
                weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'lambda':
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda x: (1. + args.gamma * float(x)) ** (-args.lr_decay))
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        if args.sdat:
            if args.lr_scheduler == 'step':
                steps = [int(step) for step in args.steps.split(',')]
                self.lr_scheduler_sam = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_sam, steps, gamma=args.gamma)
            elif args.lr_scheduler == 'exp':
                self.lr_scheduler_sam = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_sam, args.gamma)
            elif args.lr_scheduler == 'stepLR':
                steps = int(args.steps)
                self.lr_scheduler_sam = torch.optim.lr_scheduler.StepLR(self.optimizer_sam, steps, args.gamma)
            elif args.lr_scheduler == 'lambda':
                self.lr_scheduler_sam = torch.optim.lr_scheduler.LambdaLR(self.optimizer_sam, lambda x: (1. + args.gamma * float(x)) ** (-args.lr_decay))
            elif args.lr_scheduler == 'fix':
                self.lr_scheduler_sam = None
            else:
                raise Exception("lr schedule not implement")

            if args.lr_scheduler == 'step':
                steps = [int(step) for step in args.steps.split(',')]
                self.lr_scheduler_ad = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_ad, steps, gamma=args.gamma)
            elif args.lr_scheduler == 'exp':
                self.lr_scheduler_ad = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_ad, args.gamma)
            elif args.lr_scheduler == 'stepLR':
                steps = int(args.steps)
                self.lr_scheduler_ad = torch.optim.lr_scheduler.StepLR(self.optimizer_ad, steps, args.gamma)
            elif args.lr_scheduler == 'lambda':
                self.lr_scheduler_ad = torch.optim.lr_scheduler.LambdaLR(self.optimizer_ad, lambda x: (1. + args.gamma * float(x)) ** (-args.lr_decay))
            elif args.lr_scheduler == 'fix':
                self.lr_scheduler_ad = None
            else:
                raise Exception("lr schedule not implement")

        self.start_epoch = 0


        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)

        self.model_teacher.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer_teacher.to(self.device)
        self.classifier_layer_teacher.to(self.device)

        # Teacher is always in eval mode
        self.model_teacher.eval()
        if args.bottleneck:
            self.bottleneck_layer_teacher.eval()
        self.classifier_layer_teacher.eval()

        # Define the adversarial loss
        if args.domain_adversarial:
            if args.adversarial_loss == 'DA':
                self.adversarial_loss = nn.BCELoss()
            elif args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                ## add additional network for some methods
                self.softmax_layer_ad = nn.Softmax(dim=1)
                self.softmax_layer_ad = self.softmax_layer_ad.to(self.device)
                self.adversarial_loss = nn.BCELoss()
            else:
                raise Exception("loss not implement")
        else:
            self.adversarial_loss = None

        if args.loss == "cross_entropy_loss":  # label_smoothing
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss == "focal_loss":
            self.criterion = FocalLoss(gamma=args.focal_loss_gamma, reduction="mean")
        if args.self_training:
            if args.self_training_criterion == "confidence":
                if args.adaptive_confidence_threshold:
                    self.self_training_criterion = AdaptiveConfidenceBasedSelfTrainingLoss(threshold=args.confidence_threshold, num_classes=Dataset.num_classes).to(self.device)
                else:
                    self.self_training_criterion = ConfidenceBasedSelfTrainingLoss(threshold=args.confidence_threshold).to(self.device)
            elif args.self_training_criterion == "uncertainty":
                self.self_training_criterion = MCDUncertaintyBasedSelfTrainingLoss(threshold=args.confidence_threshold).to(self.device)
                self.mcd_samples = args.mcd_samples
            elif args.self_training_criterion == "oracle":
                self.self_training_criterion = OracleSelfTrainingLoss().to(self.device)
            else:
                raise Exception("Self-training criterion not implemented")
        else:
            self.self_training_criterion = None
        if args.mcc_loss:
            self.mcc_loss = MinimumClassConfusionLoss(temperature=args.mcc_temperature)

        # Set up weak-strong augmentations
        if args.use_weak_strong:
            # self.augment_weak = Compose([
            #     RandomAddGaussian(sigma=1.0),
            #     lambda t: torch.Tensor(t)
            # ])
            self.augment_weak = nn.Identity()  # random flip is already present in train_transform
            # self.augment_strong = Compose([
            #     RandomAddGaussian(sigma=1.0),
            #     #RandomCrop(crop_len=128)
            #     RandomPermutation(),
            #     RandomTimeStretch(sigma=0.3),
            #     lambda t: torch.Tensor(t)
            # ])
        else:
            self.augment_weak = nn.Identity()
            self.augment_strong = nn.Identity()

    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        step_start = time.time()
        Dataset = getattr(datasets, args.data_name)
        # ece, classwise_ece = {}, {}
        # for split in ['source_val', 'target_val']:
        #     ece[split] = ECE()
        #     classwise_ece[split] = StaticECE(n_classes=Dataset.num_classes)
        # ece_teacher, classwise_ece_teacher = {}, {}
        # for split in ['source_val', 'target_val']:
        #     ece_teacher[split] = ECE()
        #     classwise_ece_teacher[split] = StaticECE(n_classes=Dataset.num_classes)
        calibration_func = None
        optimal_temp = None

        iter_num = 0
        for epoch in range(self.start_epoch, args.max_epoch + args.additional_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, (args.max_epoch + args.additional_epoch) - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
                self.writer.add_scalar("lr/lr", self.lr_scheduler.get_lr()[0], step)
            else:
                logging.info('current lr: {}'.format(args.lr))
                self.writer.add_scalar("lr/lr", args.lr, step)

            if args.sdat:
                if self.lr_scheduler_sam is not None:
                    self.writer.add_scalar("lr/sam", self.lr_scheduler_sam.get_lr()[0], step)
                if self.lr_scheduler_ad is not None:
                    self.writer.add_scalar("lr/ad", self.lr_scheduler_ad.get_lr()[0], step)

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0.0
                epoch_acc_perclass = np.zeros(Dataset.num_classes)
                epoch_acc_teacher = 0.0
                epoch_acc_teacher_perclass = np.zeros(Dataset.num_classes)
                epoch_loss = 0.0
                epoch_loss_teacher = 0.0
                epoch_length = 0
                epoch_length_perclass = np.zeros(Dataset.num_classes, dtype="int")

                # Set models to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                    self.model_teacher.train()
                    if args.bottleneck:
                        self.bottleneck_layer_teacher.train()
                    self.classifier_layer_teacher.train()

                    with torch.no_grad():
                        # Perform one epoch on target data to update classwise accuracy based on confident pseudo-labels
                        classwise_counter = torch.zeros((Dataset.num_classes,)).to(self.device)
                        iter_target = iter(self.dataloaders['target_train'])
                        for target_inputs, target_labels in iter_target:
                            target_inputs = target_inputs.to(self.device)
                            logits = self.model_teacher_all(target_inputs)
                            # if calibration_func:
                            #     logits = calibration_func(logits)
                            if args.adaptive_confidence_threshold:
                                confidence, pseudo_labels = F.softmax(logits.detach(), dim=1).max(dim=1)
                                mask = (confidence >= self.self_training_criterion.threshold)
                                classwise_counter += pseudo_labels[mask].bincount(minlength=Dataset.num_classes)
                        if args.adaptive_confidence_threshold:
                            self.self_training_criterion.classwise_acc = classwise_counter / max(classwise_counter.max(), 1)
                            for c in range(Dataset.num_classes):
                                self.writer.add_scalar(f"adaptive_threshold/{str(c+1)}",
                                    self.self_training_criterion.classwise_acc[c], step)
                        iter_target = iter(self.dataloaders['target_train'])
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()
                    self.model_teacher.eval()
                    if args.bottleneck:
                        self.bottleneck_layer_teacher.eval()
                    self.classifier_layer_teacher.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    # Evaluation phase (source_val and target_val): no augmentation
                    if phase != 'source_train':
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    # Burn-in phase: input both strong and weak labeled source inputs into student
                    elif epoch < args.middle_epoch:
                        # if args.use_mixup:
                        #     inputs, labels = mixup(inputs, labels, args.mixup_alpha, Dataset.num_classes)
                        # if args.use_cutmix:
                        #     inputs, labels = cutmix(inputs, labels, args.mixup_alpha, Dataset.num_classes)
                        source_inputs_strong = self.augment_strong(inputs).float()
                        source_inputs_weak = self.augment_weak(inputs).float()
                        inputs = torch.cat((source_inputs_strong, source_inputs_weak), dim=0)
                        labels = torch.cat((labels, labels), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    # DA Teacher-Student mutual learning phase
                    else:
                        target_inputs, target_labels = next(iter_target)
                        # if args.use_mixup:
                        #     inputs, labels = mixup(inputs, labels, args.mixup_alpha, Dataset.num_classes)
                        #     target_inputs, target_labels = mixup(target_inputs, target_labels, args.mixup_alpha, Dataset.num_classes)
                        # if args.use_cutmix:
                        #     inputs, labels = cutmix(inputs, labels, args.mixup_alpha, Dataset.num_classes)
                        #     target_inputs, target_labels = cutmix(inputs, target_labels, args.mixup_alpha, Dataset.num_classes)
                        source_inputs_strong = self.augment_strong(inputs).float().to(self.device)
                        target_inputs_strong = self.augment_strong(target_inputs).float().to(self.device)
                        source_inputs_weak = self.augment_weak(inputs).float().to(self.device)
                        target_inputs_weak = self.augment_weak(target_inputs).float().to(self.device)
                        labels = labels.to(self.device)
                        target_labels = target_labels.to(self.device)
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # Forward
                        # Evaluation phase
                        if phase != 'source_train':
                            features = self.model(inputs)
                            if args.bottleneck:
                                features = self.bottleneck_layer(features)
                            logits = self.classifier_layer(features)
                            supervised_loss = self.criterion(logits, labels)
                            pred = logits.argmax(dim=1)
                        # Burn-in phase: supervised training of student on both strong and weak labeled source inputs
                        elif epoch < args.middle_epoch:
                            if args.sdat:
                                self.optimizer.zero_grad()
                                self.optimizer_sam.zero_grad()
                                self.optimizer_ad.zero_grad()

                                # First pass
                                features_ = self.model(inputs)
                                if args.bottleneck:
                                    features_ = self.bottleneck_layer(features_)
                                logits_ = self.classifier_layer(features_)
                                supervised_loss_ = self.criterion(logits_, labels)

                                if args.mdca_loss_source:
                                    # Calibration on source
                                    probs_ = F.softmax(logits_, dim=1)
                                    label_freq = torch.bincount(labels, minlength=Dataset.num_classes) / labels.size(0)
                                    source_mdca_loss_ = torch.abs(probs_.mean(dim=0) - label_freq).mean()
                                else:
                                    source_mdca_loss_ = 0

                                loss_ = supervised_loss_ + args.mdca_loss_weight * source_mdca_loss_

                                loss_.backward()
                                # Calculate ϵ̂ (w) and add it to the weights
                                self.optimizer_sam.first_step(zero_grad=True)

                            features = self.model(inputs)
                            if args.bottleneck:
                                features = self.bottleneck_layer(features)
                            logits = self.classifier_layer(features)
                            supervised_loss = self.criterion(logits, labels)

                            if args.mdca_loss_source:
                                # Calibration on source domain
                                probs = F.softmax(logits, dim=1)
                                label_freq = torch.bincount(labels, minlength=Dataset.num_classes) / labels.size(0)
                                source_mdca_loss = torch.abs(probs.mean(dim=0) - label_freq).mean()
                            else:
                                source_mdca_loss = 0

                            loss = supervised_loss + args.mdca_loss_weight * source_mdca_loss
                            pred = logits.argmax(dim=1)
                            acc = torch.eq(pred, labels).float().mean().item()
                        # DA Teacher-Student mutual learning phase
                        else:
                            if self.self_training_criterion:
                                # Teacher update
                                if epoch == args.self_training_epoch and batch_idx == 0:
                                    self.update_teacher(alpha=0.0)
                                else:
                                    self.update_teacher(alpha=self.alpha)

                            # Supervised training of student on both strong and weak labeled source inputs (L_sup)
                            inputs = torch.cat((source_inputs_strong, source_inputs_weak), dim=0)

                            if args.sdat:
                                self.optimizer.zero_grad()
                                self.optimizer_sam.zero_grad()
                                self.optimizer_ad.zero_grad()

                                # First pass without DA loss
                                # Forward
                                features_ = self.model(inputs)
                                if args.bottleneck:
                                    features_ = self.bottleneck_layer(features_)
                                logits_ = self.classifier_layer(features_)
                                supervised_loss_ = self.criterion(logits_, torch.cat((labels, labels), dim=0))

                                # Self-training (L_unsup)
                                if self.self_training_criterion and epoch >= args.self_training_epoch:
                                    # Pseudo-labeling of strong target inputs using teacher's predictions on weak target inputs
                                    features_ = self.model(target_inputs_strong)
                                    if args.bottleneck:
                                        features_ = self.bottleneck_layer(features_)
                                    target_logits_ = self.classifier_layer(features_)
                                    if args.self_training_criterion == "uncertainty":
                                        teacher_logits_weak_mcd_ = torch.stack([
                                            self.model_teacher_all(target_inputs_weak) for _ in range(self.mcd_samples)
                                        ], dim=1)
                                        if args.calibration and calibration_func is not None:
                                            teacher_logits_weak_mcd_ = calibration_func(teacher_logits_weak_mcd_)
                                        target_loss_, mask, pseudo_labels, confidence = self.self_training_criterion(target_logits_, teacher_logits_weak_mcd_)
                                    elif args.self_training_criterion == "oracle":
                                        teacher_logits_weak_ = self.model_teacher_all(target_inputs_weak)
                                        target_loss_, mask, pseudo_labels, confidence = self.self_training_criterion(target_logits_, teacher_logits_weak_, target_labels)
                                    else:
                                        teacher_logits_weak_ = self.model_teacher_all(target_inputs_weak)
                                        if args.calibration and calibration_func is not None:
                                            teacher_logits_weak_ = calibration_func(teacher_logits_weak_)
                                        target_loss_, mask, pseudo_labels, confidence = self.self_training_criterion(target_logits_, teacher_logits_weak_)

                                else:
                                    target_loss_ = 0

                                if args.mdca_loss_source:
                                    # Calibration on source
                                    probs_ = F.softmax(logits_, dim=1)
                                    label_freq = torch.bincount(labels, minlength=Dataset.num_classes) / labels.size(0)
                                    source_mdca_loss_ = torch.abs(probs_.mean(dim=0) - label_freq).mean()
                                else:
                                    source_mdca_loss_ = 0
                                if args.mdca_loss_target and self.self_training_criterion and epoch >= args.self_training_epoch:
                                    # Calibration on target pseudo-labels
                                    target_probs_ = F.softmax(target_logits_, dim=1)
                                    pseudo_label_freq = torch.bincount(pseudo_labels, minlength=Dataset.num_classes) / pseudo_labels.size(0)
                                    target_mdca_loss_ = torch.abs(target_probs_.mean(dim=0) - pseudo_label_freq).mean()
                                else:
                                    target_mdca_loss_ = 0

                                # Triplet loss for intra-class clustering
                                # TODO

                                if args.mcc_loss and epoch >= args.self_training_epoch:
                                    mcc_loss_ = self.mcc_loss(target_logits_)
                                else:
                                    mcc_loss_ = 0

                                loss_ = supervised_loss_ + 1.0 * target_loss_ + args.mdca_loss_weight * source_mdca_loss_ + args.mdca_loss_weight * target_mdca_loss_ + mcc_loss_

                                loss_.backward()
                                # Calculate ϵ̂ (w) and add it to the weights
                                self.optimizer_sam.first_step(zero_grad=True)

                            features = self.model(inputs)
                            if args.bottleneck:
                                features = self.bottleneck_layer(features)
                            logits = self.classifier_layer(features)
                            supervised_loss = self.criterion(logits, torch.cat((labels, labels), dim=0))
                            pred = logits.argmax(dim=1)
                            acc = torch.eq(pred, torch.cat((labels, labels), dim=0)).float().mean().item()

                            # Domain adversarial training of student on weakly augmented source and target inputs (L_dis)
                            if self.adversarial_loss is not None:
                                inputs = torch.cat((source_inputs_weak, target_inputs_weak), dim=0)
                                features = self.model(inputs)
                                if args.bottleneck:
                                    features = self.bottleneck_layer(features)
                                logits = self.classifier_layer(features)
                                if args.adversarial_loss == 'DA':
                                    domain_label_source = torch.ones(labels.size(0)).float()
                                    domain_label_target = torch.zeros(inputs.size(0)-labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_out = self.AdversarialNet(features)
                                    adversarial_loss = self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)
                                elif args.adversarial_loss == 'CDA':
                                    softmax_out = self.softmax_layer_ad(logits).detach()
                                    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    adversarial_out = self.AdversarialNet(op_out.view(-1, softmax_out.size(1) * features.size(1)))

                                    domain_label_source = torch.ones(labels.size(0)).float()
                                    domain_label_target = torch.zeros(inputs.size(0)-labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_loss = self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)
                                elif args.adversarial_loss == "CDA+E":
                                    softmax_out = self.softmax_layer_ad(logits)
                                    coeff = calc_coeff(iter_num, self.max_iter)
                                    entropy = Entropy(softmax_out)
                                    entropy.register_hook(grl_hook(coeff))
                                    entropy = 1.0 + torch.exp(-entropy)
                                    entropy_source = entropy.narrow(0, 0, labels.size(0))
                                    entropy_target = entropy.narrow(0, labels.size(0), inputs.size(0) - labels.size(0))

                                    softmax_out = softmax_out.detach()
                                    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    adversarial_out = self.AdversarialNet(
                                        op_out.view(-1, softmax_out.size(1) * features.size(1)))
                                    domain_label_source = torch.ones(labels.size(0)).float().to(
                                        self.device)
                                    domain_label_target = torch.zeros(inputs.size(0) - labels.size(0)).float().to(
                                        self.device)
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(
                                        self.device)
                                    weight = torch.cat((entropy_source / torch.sum(entropy_source).detach().item(),
                                                        entropy_target / torch.sum(entropy_target).detach().item()), dim=0)

                                    adversarial_loss = torch.sum(weight.view(-1, 1) * self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)) / torch.sum(weight).detach().item()
                                    iter_num += 1
                                else:
                                    raise Exception("loss not implement")
                                coeff = calc_coeff(self.AdversarialNet.iter_num, self.max_iter)
                                self.writer.add_scalar("grl_coeff", coeff, step)
                            else:
                                adversarial_loss = 0

                            # if args.trade_off_adversarial == 'Cons':
                            #     lam_adversarial = args.lam_adversarial
                            # elif args.trade_off_adversarial == 'Step':
                            #     lam_adversarial = 2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) /
                            #                                             (args.max_epoch-args.middle_epoch)))) - 1
                            # else:
                            #     raise Exception("loss not implement")

                            # loss = classifier_loss + lam_distance * distance_loss + lam_adversarial * adversarial_loss

                            features = self.model(target_inputs_strong)
                            if args.bottleneck:
                                features = self.bottleneck_layer(features)
                            target_logits = self.classifier_layer(features)
                            # Self-training (L_unsup)
                            if self.self_training_criterion and epoch >= args.self_training_epoch:
                                # Pseudo-labeling of strong target inputs using teacher's predictions on weak target inputs
                                if args.self_training_criterion == "uncertainty":
                                    teacher_logits_weak_mcd = torch.stack([
                                        self.model_teacher_all(target_inputs_weak) for _ in range(self.mcd_samples)
                                    ], dim=1)
                                    if args.calibration and calibration_func is not None:
                                        teacher_logits_weak_mcd = calibration_func(teacher_logits_weak_mcd)
                                    target_loss, mask, pseudo_labels, confidence = self.self_training_criterion(target_logits, teacher_logits_weak_mcd)
                                elif args.self_training_criterion == "oracle":
                                    teacher_logits_weak = self.model_teacher_all(target_inputs_weak)
                                    target_loss, mask, pseudo_labels, confidence = self.self_training_criterion(target_logits, teacher_logits_weak, target_labels)
                                else:
                                    teacher_logits_weak = self.model_teacher_all(target_inputs_weak)
                                    if args.calibration and calibration_func is not None:
                                        teacher_logits_weak = calibration_func(teacher_logits_weak)
                                    target_loss, mask, pseudo_labels, confidence = self.self_training_criterion(target_logits, teacher_logits_weak)

                                target_pred = target_logits.argmax(dim=1)
                                target_acc = torch.eq(target_pred, target_labels).float().mean().item()
                                num_pseudo_labels = mask.mean().item()
                                mean_confidence = confidence.mean().item()
                                pseudo_labels_acc = (torch.eq(pseudo_labels, target_labels)[mask.bool()]).float().mean().item()

                            else:
                                target_loss = 0

                            if args.mdca_loss_source:
                                # Calibration on source
                                probs = F.softmax(logits, dim=1)
                                label_freq = torch.bincount(labels, minlength=Dataset.num_classes) / labels.size(0)
                                source_mdca_loss = torch.abs(probs.mean(dim=0) - label_freq).mean()
                            else:
                                source_mdca_loss = 0
                            if args.mdca_loss_target and self.self_training_criterion and epoch >= args.self_training_epoch:
                                # Calibration on target pseudo-labels
                                target_probs = F.softmax(target_logits, dim=1)
                                pseudo_label_freq = torch.bincount(pseudo_labels, minlength=Dataset.num_classes) / pseudo_labels.size(0)
                                target_mdca_loss = torch.abs(target_probs.mean(dim=0) - pseudo_label_freq).mean()
                            else:
                                target_mdca_loss = 0

                            # Triplet loss for intra-class clustering
                            # TODO

                            if args.mcc_loss:
                                mcc_loss = self.mcc_loss(target_logits)
                            else:
                                mcc_loss = 0

                            loss = supervised_loss + adversarial_loss + 1.0 * target_loss + args.mdca_loss_weight * source_mdca_loss + args.mdca_loss_weight * target_mdca_loss + mcc_loss

                        if phase != 'source_train':
                            with torch.no_grad():
                                # Student evaluation
                                correct = torch.eq(pred, labels).float().sum().item()
                                epoch_loss += loss.item() * labels.size(0)
                                epoch_acc += correct
                                epoch_length += labels.size(0)
                                for c in range(Dataset.num_classes):
                                    epoch_acc_perclass[c] += ((pred == labels) * (labels == c)).float().sum().item()
                                    epoch_length_perclass[c] += (labels == c).float().sum().item()
                                # # Evaluate calibration
                                # probs = F.softmax(logits.cpu(), dim=1)
                                # # ECE
                                # ece[phase].update(probs.cpu(), labels.cpu())
                                # # class-j-ECE
                                # classwise_ece[phase].update(probs.cpu(), labels.cpu())

                                # Teacher evaluation
                                logits_teacher = self.model_teacher_all(inputs)
                                loss_teacher = self.criterion(logits_teacher, labels)
                                pred_teacher = logits_teacher.argmax(dim=1)
                                correct_teacher = torch.eq(pred_teacher, labels).float().sum().item()
                                loss_teacher_temp = loss_teacher.item() * labels.size(0)
                                epoch_loss_teacher += loss_teacher_temp
                                epoch_acc_teacher += correct_teacher
                                for c in range(Dataset.num_classes):
                                    epoch_acc_teacher_perclass[c] += ((pred_teacher == labels) * (labels == c)).float().sum().item()
                                # # Evaluate calibration
                                # probs_teacher = F.softmax(logits_teacher.cpu(), dim=1)
                                # # ECE
                                # ece_teacher[phase].update(probs_teacher.cpu(), labels.cpu())
                                # # class-j-ECE
                                # classwise_ece_teacher[phase].update(probs_teacher.cpu(), labels.cpu())

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            if args.sdat:
                                if epoch >= args.middle_epoch:
                                    self.optimizer_ad.step()
                                self.optimizer_sam.second_step(zero_grad=True)
                            else:
                                self.optimizer.step()

                            # Log the training information
                            temp_time = time.time()
                            train_time = temp_time - step_start
                            step_start = temp_time
                            # sample_per_sec = 1.0 / train_time
                            self.writer.add_scalar("loss/total_train", loss.item(), step)
                            self.writer.add_scalar("loss/supervised_train", supervised_loss.item(), step)
                            self.writer.add_scalar("accuracy/source_train", acc, step)
                            self.writer.add_scalar("time", train_time, step)
                            if self.adversarial_loss and epoch >= args.middle_epoch:
                                self.writer.add_scalar("loss/adversarial_train", adversarial_loss.item(), step)
                            if self.self_training_criterion and epoch >= args.self_training_epoch:
                                self.writer.add_scalar("loss/target_train", target_loss.item(), step)
                                self.writer.add_scalar("accuracy/target_train", target_acc, step)
                                self.writer.add_scalar("pseudo_labels/num_pseudo_labels", num_pseudo_labels, step)
                                self.writer.add_scalar("pseudo_labels/average_confidence", mean_confidence, step)
                                self.writer.add_scalar("pseudo_labels/accuracy", pseudo_labels_acc, step)
                                self.writer.add_histogram("pseudo_labels/confidence", confidence, step)
                                self.writer.add_histogram("pseudo_labels/pseudo_labels", pseudo_labels, step)
                                # logging.error(self.self_training_criterion.classwise_acc)
                                if args.mdca_loss_target:
                                   self.writer.add_scalar("loss/target_mdca", target_mdca_loss.item(), step)
                            if args.mdca_loss_source:
                                self.writer.add_scalar("loss/source_mdca", source_mdca_loss.item(), step)
                            if args.mcc_loss and epoch >= args.middle_epoch:
                                self.writer.add_scalar("loss/mcc", mcc_loss.item(), step)
                            step += 1
                            if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                                break

                if phase != 'source_train':
                    # Print the val information via each epoch
                    epoch_loss = epoch_loss / epoch_length
                    epoch_acc = epoch_acc / epoch_length
                    epoch_acc_perclass = epoch_acc_perclass / epoch_length_perclass
                    logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                        epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                    ))
                    self.writer.add_scalar(f"loss/{phase}", epoch_loss, step)
                    self.writer.add_scalar(f"accuracy/{phase}", epoch_acc, step)
                    self.writer.add_scalar(f"balanced_accuracy/{phase}", np.mean(epoch_acc_perclass), step)
                    for c in range(Dataset.num_classes):
                        self.writer.add_scalar(f"accuracy_perclass/{phase}_{str(c+1)}", epoch_acc_perclass[c], step)
                    # self.writer.add_scalar(f"calibration/ECE_{phase}", ece[phase].score(), step)
                    # classwise_ece_scores = classwise_ece[phase].score()
                    # for c in range(Dataset.num_classes):
                    #     self.writer.add_scalar(f"calibration/class-{str(c+1)}-CE_{phase}", classwise_ece_scores[c], step)
                    # self.writer.add_scalar(f"calibration/SCE_{phase}", classwise_ece_scores.mean(), step)

                    if self.self_training_criterion:
                        epoch_loss_teacher = epoch_loss_teacher / epoch_length
                        epoch_acc_teacher = epoch_acc_teacher / epoch_length
                        epoch_acc_teacher_perclass = epoch_acc_teacher_perclass / epoch_length_perclass
                        logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec (teacher)'.format(
                            epoch, phase, epoch_loss_teacher, phase, epoch_acc_teacher, time.time() - epoch_start
                        ))
                        self.writer.add_scalar(f"loss/{phase}_teacher", epoch_loss_teacher, step)
                        self.writer.add_scalar(f"accuracy/{phase}_teacher", epoch_acc_teacher, step)
                        for c in range(Dataset.num_classes):
                            self.writer.add_scalar(f"accuracy_perclass/{phase}_teacher_{str(c+1)}", epoch_acc_teacher_perclass[c], step)
                        # self.writer.add_scalar(f"calibration/ECE_{phase}_teacher", ece_teacher[phase].score(), step)
                        # classwise_ece_scores_teacher = classwise_ece_teacher[phase].score()
                        # for c in range(Dataset.num_classes):
                        #     self.writer.add_scalar(f"calibration/class-{str(c+1)}-CE_{phase}_teacher", classwise_ece_scores_teacher[c], step)
                        # self.writer.add_scalar(f"calibration/SCE_{phase}_teacher", classwise_ece_scores_teacher.mean(), step)

                    # save the model
                    if args.save_weights and phase == 'target_val':
                        # save the best model according to the val accuracy
                        if (epoch_acc > best_acc or epoch > (args.max_epoch + args.additional_epoch)-2) and (epoch > args.middle_epoch-1):
                            best_acc = epoch_acc
                            logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                            # save the checkpoints
                            if self.self_training_criterion:
                                model_state_dic = self.model_teacher_all.state_dict()
                                torch.save(model_state_dic,
                                    os.path.join(self.save_dir, '{}-{:.4f}-best_model_teacher.pth'.format(epoch, best_acc)))
                            model_state_dic = self.model_all.state_dict()
                            torch.save(model_state_dic,
                                os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            if args.sdat:
                if self.lr_scheduler_ad is not None:
                    self.lr_scheduler_ad.step()
                if self.lr_scheduler_sam is not None:
                    self.lr_scheduler_sam.step()
                if epoch < args.middle_epoch and self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            """Evaluate and re-calibrate the model."""
            self.model_teacher_all.eval()
            self.model.eval()
            if args.bottleneck:
                self.bottleneck_layer.eval()
            if args.domain_adversarial:
                self.AdversarialNet.eval()
            self.classifier_layer.eval()
            calibration_func, optimal_temp, ece_scores_teacher = get_optimal_temp(
                args.calibration if epoch >= args.calibration_epoch else None,
                self.dataloaders,
                nn.Sequential(self.model_teacher, self.bottleneck_layer_teacher),
                self.classifier_layer_teacher,
                output_name=args.data_name,
                previous_temp=optimal_temp,
                alpha=0.9 if args.temperature_ema else 0.0
            )
            _, _, ece_scores = get_optimal_temp(
                None,
                self.dataloaders,
                nn.Sequential(self.model, self.bottleneck_layer),
                self.classifier_layer,
                output_name=args.data_name
            )
            if optimal_temp is not None:
                self.writer.add_scalar("calibration/optimal_temp", optimal_temp, step)
            for metric_name, metric_value in ece_scores.items():
                self.writer.add_scalar(f"calibration/{metric_name}", metric_value, step)
            for metric_name, metric_value in ece_scores_teacher.items():
                self.writer.add_scalar(f"calibration/{metric_name}_teacher", metric_value, step)

        if args.dump_features:
            """Save source_val and target features, logits and labels for TransCal."""
            features_dir = os.path.join(self.save_dir, "features")
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)
            generate_feature_wrapper(
                self.dataloaders,
                nn.Sequential(self.model, self.bottleneck_layer),
                self.classifier_layer,
                features_dir,
                output_name=args.data_name + "_student"
            )
            generate_feature_wrapper(
                self.dataloaders,
                nn.Sequential(self.model_teacher, self.bottleneck_layer_teacher),
                self.classifier_layer_teacher,
                features_dir,
                output_name=args.data_name + "_teacher"
            )
