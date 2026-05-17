import os
import yaml
import math
import random
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from typing import Optional

import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed

import argparse
from thop import profile

from models import resnet, vggsnn
from models.submodules.layers import LIF, ASLIF, IF, ASIF, ReLU, MPBNLIF
from models.submodules.layers import Conv, ConvA, ConvN, ConvBN, ConvTEBN, ConvBNTT
from models.submodules.initialization import _calculate_fan_in
from utils.augment import CIFAR10Policy, ImageNetPolicy, Cutout, DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.utils import RecordDict, GlobalTimer, Timer, str2bool, count_conv2d, count_linear
from utils.utils import DatasetSplitter, DatasetWarpper, CriterionWarpper, DVStransform, SOPMonitor
from utils.utils import is_main_process, save_on_master, tb_record, accuracy, safe_makedirs
from spikingjelly.activation_based import functional, layer, base, monitor
from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model


def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Training')

    # dataset options
    parser.add_argument('--dataset', default='CIFAR10', help='dataset type')
    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--input-size', default=(3, 32, 32), type=int, nargs='+')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num-workers', default=16, type=int)

    parser.add_argument('--advanced-aug', type=str2bool, help='advanced augmentation')
    parser.add_argument('--no-aug', type=str2bool, help='no augmentation')
    parser.add_argument('--re-prob', default=0.0, type=float, help='random erasing prob')
    parser.add_argument('--re-mode', default='pixel', type=str, help='random erasing mode')
    parser.add_argument('--re-count', default=1, type=int, help='random erasing count')
    parser.add_argument('--re-split', default=False, type=bool, help='random erasing split')
    parser.add_argument('--scale', default=[0.08, 1.0], type=float, nargs='+',
                        help='input re-scale')
    parser.add_argument('--ratio', default=[3.0 / 4.0, 4.0 / 3.0], type=float, nargs='+',
                        help='input re-ratio')
    parser.add_argument('--hflip', default=0.5, type=float, help='horizontal flip prob')
    parser.add_argument('--vflip', default=0.0, type=float, help='vertical flip prob')
    parser.add_argument('--color-jitter', default=0.4, type=float, help='color jitter')
    parser.add_argument('--auto-augment', default='rand-m9-mstd0.5-inc1', type=str,
                        help='auto augment policy')
    parser.add_argument('--num-aug-repeats', default=0, type=int, help='auto augment repeat')
    parser.add_argument('--num-aug-splits', default=0, type=int, help='auto augment split')
    parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation mode')
    parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=float, nargs='+')
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=float, nargs='+')
    parser.add_argument('--crop-pct', default=0.875, type=float)
    parser.add_argument('--crop-pct-eval', default=None, type=float)
    parser.add_argument('--crop-mode', default='random', type=str)

    parser.add_argument('--mixup-alpha', default=0.0, type=float)
    parser.add_argument('--cutmix-alpha', default=0.0, type=float)
    parser.add_argument('--cutmix-minmax', default=None, type=float, nargs='+')
    parser.add_argument('--mixup-prob', default=1.0, type=float)
    parser.add_argument('--mixup-switch-prob', default=0.5, type=float)
    parser.add_argument('--mixup-mode', default='batch', type=str)
    parser.add_argument('--label-smoothing', default=0, type=float)

    parser.add_argument('--dvs-augment', action='store_true', help='DVS augment')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='vgg11snn', help='model type')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', type=str,
                        help='learning rate scheduler')
    parser.add_argument('--cooldown-epochs', default=0, type=int, help='cooldown epochs')
    parser.add_argument('--min-lr', default=0, type=float, help='minimum learning rate')
    parser.add_argument('--warmup-lr', default=0, type=float, help='warmup learning rate')
    parser.add_argument('--warmup-epochs', default=0, type=int, help='warmup epochs')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--accumulation-steps', default=1, type=int,
                        help='gradient accumulation steps')

    # other options
    parser.add_argument('--output-path', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=str2bool, default=True, help='Use AMP training')

    parser.add_argument('--print-freq', default=5, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--tb-interval', type=int, default=10)
    parser.add_argument('--distributed-init-mode', type=str, default='env://')
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")

    # ablation

    parser.add_argument('--conv', type=str, default='Conv')
    parser.add_argument('--activation', type=str, default='ASLIF')
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--init-method', type=str, default='spiking')
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--decay-input', type=str2bool, default=True)
    parser.add_argument('--p-init', type=float, default=0.1)
    parser.add_argument('--zero-init-residual', type=str2bool, default=True)

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args


def setup_logger(output_path):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_path, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def load_data(
    dataset_dir: str,
    dataset_type: str,
    num_classes: int,
    distributed: bool,
    args: argparse.Namespace,
):

    if dataset_type == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                                     download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
    elif dataset_type == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
    elif dataset_type in ['ImageNet', 'ImageNet100', 'TinyImageNet']:
        dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'))
        dataset_test = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'))
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=args.T,
                             split_by='number')
        dataset_train, dataset_test = DatasetSplitter(dataset, 0.9,
                                                      True), DatasetSplitter(dataset, 0.1, False)
    elif dataset_type == 'DVS128Gesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame',
                                      frames_number=args.T, split_by='number')
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame',
                                     frames_number=args.T, split_by='number')
    else:
        raise ValueError(dataset_type)

    if args.advanced_aug:
        if args.mixup_alpha > 0. or args.cutmix_alpha > 0. or args.cutmix_minmax is not None:
            collate_fn = FastCollateMixup(
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.label_smoothing,
                num_classes=num_classes,
            )
        else:
            collate_fn = None
        data_loader_train = create_loader(
            dataset_train,
            input_size=args.input_size,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=args.no_aug,
            re_prob=args.re_prob,
            re_mode=args.re_mode,
            re_count=args.re_count,
            re_split=args.re_split,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.auto_augment,
            num_aug_repeats=args.num_aug_repeats,
            num_aug_splits=args.num_aug_splits,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            num_workers=args.num_workers,
            distributed=distributed,
            crop_pct=args.crop_pct,
            crop_mode=args.crop_mode,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=args.input_size,
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            num_workers=args.num_workers,
            distributed=distributed,
            crop_pct=args.crop_pct_eval,
            pin_memory=True,
        )
    else:
        if dataset_type == 'CIFAR10':
            if not args.no_aug:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    Cutout(n_holes=1, length=16), ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), ])

        elif dataset_type == 'CIFAR100':
            if not args.no_aug:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                         std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                    Cutout(n_holes=1, length=8), ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                         std=[n / 255. for n in [68.2, 65.4, 70.4]]), ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]]), ])
        elif dataset_type in ['ImageNet', 'ImageNet100']:
            if not args.no_aug:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(
                        args.input_size[-2:], antialias=True,
                        interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])
            else:
                transform_train = transforms.Compose([
                    transforms.Resize(args.input_size[-2:], antialias=True,
                                      interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])
            transform_test = transforms.Compose([
                transforms.Resize(
                    (int(args.input_size[-2] / 0.875), int(args.input_size[-1] / 0.875)),
                    antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(args.input_size[-2:]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])
        else:
            if not args.no_aug:
                transform_train = DVStransform(transform=transforms.Compose([
                    transforms.Resize(size=args.input_size[-2:], antialias=True,
                                      interpolation=transforms.InterpolationMode.BICUBIC),
                    DVSAugment()]))
            else:
                transform_train = DVStransform(transform=transforms.Compose([
                    transforms.Resize(size=args.input_size[-2:], antialias=True,
                                      interpolation=transforms.InterpolationMode.BICUBIC)]))
            transform_test = DVStransform(
                transform=transforms.Resize(size=args.input_size[-2:], antialias=True,
                                            interpolation=transforms.InterpolationMode.BICUBIC))

        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                       sampler=test_sampler,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True, drop_last=False)

    return dataset_train, dataset_test, data_loader_train, data_loader_test


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    print_freq: int,
    factor: int,
    scheduler_per_iter: Optional[BaseSchedulerPerIter] = None,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1,
    one_hot=None,
):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]

    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            image, target = image.float().cuda(), target.cuda()
            if scaler is not None:
                with autocast('cuda'):
                    output = model(image)
                    if one_hot:
                        loss = criterion(output, F.one_hot(target, one_hot).float())
                    else:
                        loss = criterion(output, target)
            else:
                output = model(image)
                if one_hot:
                    loss = criterion(output, F.one_hot(target, one_hot).float())
                else:
                    loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()  # type:ignore
            else:
                loss.backward()

            if (idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if scheduler_per_iter is not None:
                scheduler_per_iter.step()

            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1_s, batch_size)
            metric_dict['acc@5'].update(acc5_s, batch_size)

        if print_freq != 0 and ((idx + 1) % math.ceil(len(data_loader_train) / (print_freq))) == 0:
            #torch.distributed.barrier()
            metric_dict.sync()
            logger.debug(' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                idx + 1, len(data_loader_train),
                (idx + 1) * batch_size * factor / timer_container[0], metric_dict['loss'].ave,
                metric_dict['acc@1'].ave, metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, one_hot):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    mon = monitor.OutputMonitor(model, (LIF, ASLIF, IF, ASIF), lambda x: x.mean().unsqueeze(0))
    mon.enable()
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.float().to(torch.device('cuda'), non_blocking=True)
            target = target.to(torch.device('cuda'), non_blocking=True)
            output = model(image)
            if one_hot:
                loss = criterion(output, F.one_hot(target, one_hot).float())
            else:
                loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % math.ceil(len(data_loader) / print_freq)) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    firing_rate_dict = {}
    for name in mon.monitored_layers:
        sublist = mon[name]
        firing_rate_dict[name] = torch.cat(sublist).mean().item()
        logger.debug('Layer: {}, firing rate: {:.5f}'.format(name, firing_rate_dict[name]))
    for name, module in model.named_modules():
        if isinstance(module, ASLIF):
            logger.debug('ASLIF {}: gamma: {:.5f}, beta: {:.5f}'.format(
                name,
                module.gamma.mean().item(),
                module.beta.mean().item()))
    mon.disable()
    mon.remove_hooks()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict[
        'acc@5'].ave, firing_rate_dict


def test(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    logger: logging.Logger,
):

    safe_makedirs(os.path.join(args.output_path, 'test'))
    mon = SOPMonitor(model)

    logger.info('[Test]')

    model.eval()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image).mean(0)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if args.print_freq != 0 and ((idx + 1) %
                                         math.ceil(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))

    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    for name, module in model.named_modules():
        if isinstance(module, ASLIF):
            logger.info('ASLIF {}: gamma: {:.5f}, beta: {:.5f}'.format(
                name,
                module.gamma.mean().item(),
                module.beta.mean().item()))
    step_mode = 's'
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break

    ops, params = profile(
        model, inputs=(inputs, ), verbose=False, custom_ops={
            layer.Conv2d: count_conv2d,
            Conv: count_conv2d,
            ConvA: count_conv2d,
            ConvN: count_conv2d,
            ConvBN: count_conv2d,
            ConvTEBN: count_conv2d,
            ConvBNTT: count_conv2d,
            layer.Linear: count_linear})[0:2]
    if step_mode == 'm':
        ops, params = (ops / (1000**3)) / args.T, params / (1000**2)
    else:
        ops, params = (ops / (1000**3)), params / (1000**2)
    functional.reset_net(model)
    logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop
    sops = sops / (1000**3)
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / args.batch_size
    if step_mode == 's':
        sops = sops * args.T
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops + 1e-10)))


def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    safe_makedirs(args.output_path)
    logger = setup_logger(args.output_path)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # load data

    dataset_type = args.dataset
    if dataset_type == 'CIFAR10':
        num_classes = 10
        one_hot = 10
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        one_hot = 100
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        one_hot = None
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'ImageNet100':
        num_classes = 100
        one_hot = 100
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'TinyImageNet':
        num_classes = 200
        one_hot = None
        inputs = torch.rand(1, *args.input_size).cuda()
    elif dataset_type == 'CIFAR10DVS':
        num_classes = 10
        one_hot = 10
        inputs = torch.rand(1, 1, *args.input_size).cuda()
    elif dataset_type == 'DVS128Gesture':
        num_classes = 11
        one_hot = 11
        inputs = torch.rand(1, 1, *args.input_size).cuda()
    else:
        raise ValueError(dataset_type)

    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, dataset_type, num_classes, distributed, args)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    # model

    ckwargs = {}
    if args.conv is not None:
        if args.conv == 'Conv':
            ckwargs['conv'] = Conv
        elif args.conv == 'ConvA':
            ckwargs['conv'] = ConvA
        elif args.conv == 'ConvN':
            ckwargs['conv'] = ConvN
        elif args.conv == 'ConvBN':
            ckwargs['conv'] = ConvBN
        elif args.conv == 'ConvTEBN':
            ckwargs['conv'] = ConvTEBN
        elif args.conv == 'ConvBNTT':
            ckwargs['conv'] = ConvBNTT
        else:
            raise ValueError(args.conv)
    if args.activation is not None:
        if args.activation == 'LIF':
            ckwargs['activation'] = LIF
        elif args.activation == 'ASLIF':
            ckwargs['activation'] = ASLIF
        elif args.activation == 'IF':
            ckwargs['activation'] = IF
        elif args.activation == 'ASIF':
            ckwargs['activation'] = ASIF
        elif args.activation == 'MPBNLIF':
            ckwargs['activation'] = MPBNLIF
        elif args.activation == 'ReLU':
            ckwargs['activation'] = ReLU
        else:
            raise ValueError(args.activation)
    if args.init_method is not None:
        ckwargs['init_method'] = args.init_method
    if args.p_init is not None:
        ckwargs['p_init'] = args.p_init
    if args.zero_init_residual is not None:
        ckwargs['zero_init_residual'] = args.zero_init_residual
    conv_kwargs = {}
    if args.bias is not None:
        conv_kwargs['bias'] = args.bias
    if args.conv == 'ConvTEBN' or args.conv == 'ConvBNTT':
        conv_kwargs['T'] = args.T
    if args.conv == 'ConvBN' and args.threshold is not None:
        # calc lambda
        lmbda = 1.0
        if args.activation in ['LIF', 'ASLIF', 'MPBNLIF'] and args.decay_input:
            lmbda = 1.0 / args.tau
        conv_kwargs['threshold'] = args.threshold / lmbda
    if len(conv_kwargs) > 0:
        ckwargs['conv_kwargs'] = conv_kwargs
    activation_kwargs = {}
    if args.tau is not None:
        activation_kwargs['tau'] = args.tau
    if args.threshold is not None:
        activation_kwargs['v_threshold'] = args.threshold
    if args.decay_input is not None:
        activation_kwargs['decay_input'] = args.decay_input
    if len(activation_kwargs) > 0:
        ckwargs['activation_kwargs'] = activation_kwargs

    model = create_model(
        args.model,
        num_classes=num_classes,
        in_channels=args.input_size[0],
        input_size=args.input_size[1],
        T=args.T,
        **ckwargs,
    )

    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimzer

    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # loss_fn

    if args.mixup_alpha > 0. or args.cutmix_alpha > 0. or args.cutmix_minmax is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)
    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = CriterionWarpper(criterion_eval)

    # amp speed up

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # lr scheduler

    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=args.lr_scheduler,
        num_epochs=args.epochs,
        cooldown_epochs=args.cooldown_epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
    )

    # DDP

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module

    # custom scheduler

    scheduler_per_iter = None
    scheduler_per_epoch = None

    # resume

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        # custom scheduler
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if is_main_process():
            test(model_without_ddp, data_loader_test, inputs, args, logger)
        return

    ##################################################
    #                   Train
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_path, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch, optimizer.param_groups[0]["lr"]))

        with Timer(' Train', logger):
            train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer,
                                                                 data_loader_train, logger,
                                                                 args.print_freq, world_size,
                                                                 scheduler_per_iter, scaler,
                                                                 args.accumulation_steps, one_hot)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        with Timer(' Test', logger):
            test_loss, test_acc1, test_acc5, firing_rate_dict = evaluate(
                model, criterion_eval, data_loader_test, args.print_freq, logger, one_hot)

        gamma_dict = {}
        beta_dict = {}
        weight_std_dict = {}
        for name, module in model_without_ddp.named_modules():
            if isinstance(module, (ASLIF, ASIF)):
                gamma_dict[name] = module.gamma.mean().item()
                beta_dict[name] = module.beta.mean().item()
        for name, module in model_without_ddp.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_std_dict[name] = module.weight.std().item() * _calculate_fan_in(
                    module.weight)**0.5
        if is_main_process() and tb_writer is not None:
            tb_record(tb_writer, train_loss, train_acc1, train_acc5, test_loss, test_acc1,
                      test_acc5, firing_rate_dict, gamma_dict, beta_dict, weight_std_dict, epoch)

        logger.info(' Test loss: {:.5f}, Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_loss, test_acc1, test_acc5))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': test_acc1 if max_acc1 < test_acc1 else max_acc1, }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        # custom scheduler

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_path, 'checkpoint_latest.pth'))

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_on_master(checkpoint, os.path.join(args.output_path, 'checkpoint_max_acc1.pth'))

    logger.info('Training completed.')

    ##################################################
    #                   test
    ##################################################

    ##### reset utils #####

    # reset model

    del model, model_without_ddp

    model = create_model(
        args.model,
        num_classes=num_classes,
        in_channels=args.input_size[0],
        input_size=args.input_size[1],
        T=args.T,
        **ckwargs,
    ).cuda()

    try:
        checkpoint = torch.load(os.path.join(args.output_path, 'checkpoint_max_acc1.pth'),
                                map_location='cpu')
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    model.load_state_dict(checkpoint['model'])

    # reload data

    del dataset_train, dataset_test, data_loader_train, data_loader_test

    _, _, _, data_loader_test = load_data(args.data_path, dataset_type, num_classes, False, args)

    ##### test #####

    if is_main_process():
        test(model, data_loader_test, inputs, args, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()
