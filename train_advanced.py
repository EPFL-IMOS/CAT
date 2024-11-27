#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils
import torch
import warnings
print(torch.__version__)
warnings.filterwarnings('ignore')
import numpy as np
import random

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--model_name', type=str, default='cnn_features_1d', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='PHMFFT', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='D:\Data\PHM2009gearbox\PHM_Society_2009_Competition_Expanded_txt', help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[0], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument("--save_weights", action='store_true', default=False, help='whether to save model weights')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')

    # classification loss
    parser.add_argument('--loss', type=str, choices=['cross_entropy_loss', 'focal_loss'], default='cross_entropy_loss', help='the classification loss')
    parser.add_argument('--focal_loss_gamma', type=float, default=2, help='focal loss exponent gamma')

    #
    parser.add_argument('--distance_metric', type=bool, default=False, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD', 'JMMD', 'CORAL'], default='MK-MMD', help='which distance loss you use')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')

    #
    parser.add_argument('--domain_adversarial', type=bool, default=False, help='whether use domain_adversarial')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA', 'CDA+E'], default='CDA+E', help='which adversarial loss you use')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')
    #
    parser.add_argument('--self_training', type=bool, default=False, help='whether to use self-training with pseudo-labels')
    parser.add_argument('--self_training_criterion', type=str, choices=['confidence', 'uncertainty', 'oracle'], default='confidence', help='criterion to select pseudo-labels')
    parser.add_argument('--self_training_epoch', type=int, default=50, help='epoch to start self-training')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='threshold value on confidence to select pseudo-labels')
    parser.add_argument('--adaptive_confidence_threshold', default=False, action="store_true", help='adaptive per-class confidence threshold')
    parser.add_argument('--mcd_samples', type=int, default=10, help='number of Monte-Carlo Dropout inferences')
    parser.add_argument('--lam_self_training', type=float, default=1, help='this is used for Cons')
    #
    parser.add_argument('--mdca_loss_source', default=False, action="store_true", help='MDCA calibration loss on source')
    parser.add_argument('--mdca_loss_target', default=False, action="store_true", help='MDCA calibration loss on target (pseudo-labels)')
    parser.add_argument('--mdca_loss_weight', type=float, default=10, help='weight of MDCA loss')
    #
    parser.add_argument('--mcc_loss', default=False, action="store_true", help='MCC (Minimum Class Confusion) loss on target predictions')
    parser.add_argument('--mcc_temperature', type=float, default=2.0, help='temperature for scaling in MCC loss')

    # cutmix
    parser.add_argument('--use_cutmix', action='store_true', default=False)

    # mixup
    parser.add_argument('--use_mixup_source', action='store_true', default=False)
    parser.add_argument('--use_mixup_target', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', type=float, default=1)

    # manifold mixup
    parser.add_argument('--use_manifold_mixup_source', action='store_true', default=False)
    parser.add_argument('--use_manifold_mixup_target', action='store_true', default=False)
    parser.add_argument('--manifold_mixup_alpha', type=float, default=1)

    # domain mixup
    parser.add_argument('--use_domain_mixup', action='store_true', default=False)
    parser.add_argument('--domain_mixup_alpha', type=float, default=1)

    # domain manifold mixup
    parser.add_argument('--use_domain_manifold_mixup', action='store_true', default=False)
    parser.add_argument('--domain_manifold_mixup_alpha', type=float, default=1)

    # remix
    # parser.add_argument('--use_remix', action='store_true')

    # calibration
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument('--calibration_epoch', type=int, default=150, help='epoch to start calibration')
    parser.add_argument("--temperature_ema", action='store_true', default=False)

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix', 'lambda'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lambda lr scheduler')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')
    parser.add_argument('--sdat', action='store_true', default=False, help="use SDAT (Smooth Domain-Adversarial Training)")

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='fix maximum number of iterations per epoch')
    # parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')

    parser.add_argument('--seed', type=int, default=42, help='seed')

    parser.add_argument('--dump_features', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # Seed everything
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    # Prepare the saving path for the model
    #sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    #save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    save_dir = args.checkpoint_dir + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
