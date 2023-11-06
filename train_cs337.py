import os, sys, time, torch, random, argparse, json, copy
import itertools
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.distributions import Categorical
import datetime, pytz
from typing import Type, Any, Callable, Union, List, Optional
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
from pathlib import Path

import matplotlib.pyplot as plt

from model_dict import get_model_from_name
from utils import get_model_infos
from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time
from starts import prepare_logger, prepare_seed
from get_dataset_with_transform import get_datasets
import torch.utils.data as data
from DiSK import obtain_accuracy, get_mlr, save_checkpoint, evaluate_model

def m__get_prefix( args ):
    prefix = args.file_name + '_' + args.dataset + '-' + args.model_name
    return prefix

def get_model_prefix( args ):
    prefix = os.path.join(args.save_dir, m__get_prefix( args ))
    return prefix

# used just for evaluation
def cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, xloader, criterion, batch_size, mode='eval' ):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if mode == 'eval': 
        network.eval()
    else:
        network.train()

    progress = ProgressMeter(
            logger,
            len(xloader),
            [losses, top1, top5],
            prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets) in enumerate(xloader):
        if mode == 'train':
            optimizer.zero_grad()

        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        _, logits, _ = network(inputs)

        loss = criterion(logits, targets)
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        if mode == 'train':
            loss.backward()
            optimizer.step()

        losses.update(loss.mean().item(), inputs.size(0))
        top1.update(prec1.mean().item(), inputs.size(0))
        top5.update(prec5.mean().item(), inputs.size(0))

        if mode == 'train':
            scheduler.step(epoch)

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                progress.display(i)

    return losses.avg, top1.avg, top5.avg

def main(args):
    rand_seeds = [args.rand_seed1, args.rand_seed2, args.rand_seed3]
    k=len(rand_seeds)
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    criterion = nn.CrossEntropyLoss()
    
    dataset=args.dataset
    if dataset=="cifar100" or dataset=="cifar10":
        
        train_data, test_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
        )
        train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//10, 
                                                            len(train_data)//10])
    
    
    else:
        train_data, test_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
        )
        train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//5,  len(train_data)//5])
        test_data, valid_data = data.random_split(valid_data, [len(valid_data)-len(valid_data)//2,  len(valid_data)//2])
    
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    args.class_num = class_num
    logger = prepare_logger(args)
    prepare_seed(args.rand_seed)
    
    logger.log(args.__str__())
    logger.log("Train:{}\t, Valid:{}\t, Test:{}\n".format(len(train_data),
                                                          len(valid_data),
                                                          len(test_data)))
    Arguments = namedtuple("Configure", ('class_num','dataset','dropout')  )
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset,'dropout':0 }
    model_config = Arguments(**md_dict)


    # STUDENT
    base_model = [i for i in range(k)]
    for i in range(k):
        base_model[i] = get_model_from_name( model_config, args.model_name )
        logger.log("Student {}:".format(args.model_name) )
        model_name = args.model_name

    # ce_ptrained_path = "./pretrained/disk-CE-cifar100-ResNet10_s-model_best.pth.tar"
    ce_pretrained_path = [i for i in range(k)]
    for i in range(k):
        ce_ptrained_path = "../models/ce_results_tinyImagenet/CE_with_seed-{}_cycles-1-_{}_{}-{}_"\
                            "model_best.pth.tar".format(args.rand_seed[i],
                                                        #args.sched_cycles,
                                                        args.model_name,
                                                        args.dataset,
                                                        args.model_name)


    logger.log("using pretrained student model from {}".format(ce_ptrained_path))

    network = [i for i in range(k)]
    best_state_dict = [i for i in range(k)]
    base_model = [i for i in range(k)]
    for i in range(k):
        if args.pretrained_student: # load CE-pretrained student
            assert Path().exists(), "Cannot find the initialization file : {:}".format(ce_ptrained_path)
            base_checkpoint = torch.load(ce_ptrained_path)
            base_model[i].load_state_dict(base_checkpoint["base_state_dict"])
        
        
        base_model[i] = base_model[i].cuda()
        network[i] = base_model[i] 
        best_state_dict[i] = copy.deepcopy( base_model.state_dict() )
    #testing pretrained student
    
    for i in range(k):
        test_loss, test_acc1, test_acc5 = evaluate_model( network[i], test_loader, criterion, args.eval_batch_size )
        logger.log(
            "***{:s}*** before training [Student(CE)]  Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
                time_string(),
                test_loss,
                test_acc1,
                test_acc5,
                100 - test_acc1,
                100 - test_acc5,
            )
            )
        # set student training up
    
    optimizer_s = [i for i in range(k)]
    scheduler_s = [i for i in range(k)]
    for i in range(k):
        optimizer_s[i] = torch.optim.SGD(base_model[i].parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler_s[i] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_s[i], args.epochs//args.sched_cycles)
        logger.log("Scheduling LR update to student no {}, {} time at {}-epoch intervals".format(k,args.sched_cycles, 
                                                                                        args.epochs//args.sched_cycles))

    # TEACHER
    Teacher_model = get_model_from_name( model_config, args.teacher )
    # PATH="/home/shashank/disk/model_10l/disk-CE-cifar100-ResNet10_l-model_best.pth.tar"
    #/home/anmolreddy/pretrained/teacher_seed-1_cifar100-ResNet10_lmodel_best.pth.tar
    teach_PATH = "../models/ce_results_tinyImagenet/CE_with_seed-{}_cycles-1-_{}_{}-{}_"\
                    "model_best.pth.tar".format(args.rand_seed,
                                                args.teacher,
                                                args.dataset,
                                                args.teacher)
    teach_checkpoint = torch.load(teach_PATH)
    Teacher_model.load_state_dict(teach_checkpoint['base_state_dict'])
    Teacher_model = Teacher_model.cuda()
    network_t = Teacher_model
    network_t.eval()

    #testing teacher
    test_loss, test_acc1, test_acc5 = evaluate_model( network_t, test_loader, nn.CrossEntropyLoss(), args.eval_batch_size )
    logger.log(
        "***{:s}*** [Teacher] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )

    flop, param = get_model_infos(base_model, xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "[Student]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )

    logger.log("-" * 50)

    best_acc, best_epoch = 0.0, 0
    log_file_name = get_model_prefix( args )

    for m_idx in range(0, 3):
        m = network[m_idx]
        for m_ in network[:m_idx]:
            m_.eval()
        for epoch in range(args.epochs):
            mode='train'
            logger.log("\nStarted EPOCH:{}".format(epoch))

            losses =  AverageMeter('Loss', ':.4e') 
            top1 =  AverageMeter('Acc@1', ':6.2f') 
            top5 =  AverageMeter('Acc@5', ':6.2f')

            
            base_model[m_idx].train()
            progress = ProgressMeter(
                        logger,
                        len(train_loader),
                        [losses, top1, top5],
                        prefix="[{}] E: [{}]".format(mode.upper(), epoch))
                
            for iteration, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)

                features, logits, _ = network(inputs)
                with torch.no_grad():
                    _,teacher_logits , _ = network_t(inputs)

                T=args.temperature

                log_student = F.log_softmax(logits / T, dim=1)
                    
                sof_teacher = F.softmax(teacher_logits / T, dim=1)
                
                alpha = args.loss_kd_frac
                optimizer_s.zero_grad()
                loss_kd = nn.KLDivLoss(reduction = "batchmean")(log_student, sof_teacher) * T * T
                loss_ce =  F.cross_entropy(logits, targets)
                loss = loss_ce

                ## DBAT Loss ##
                if m_idx != 0:                        
                    p_1_s, indices = [], []
                    # print(ensemble)
                    with torch.no_grad():
                        for m_ in network[:m_idx]:
                            # print("hello")
                            # print(m_(inputs,lengths).shape)
                            p_1 = torch.softmax(m_(inputs), dim=-1)
                            p_1, idx = p_1.max(dim=-1)
                            p_1_s.append(p_1)
                            indices.append(idx)
                    # print(p_1_s[0].shape)
                    
                    p_2 = torch.softmax(m(inputs), dim=-1)
                    # print(p_2.shape)
                    # print(indices[0].shape)
                    p_2 = torch.reshape(p_2, (-1,p_2.shape[-1]))
                    p_2_s = [p_2[torch.arange(torch.ravel(max_idx).shape[0]), torch.ravel(max_idx)] for max_idx in indices]
                    # print("P 2 s shape", p_2_s[0].shape)
                    # Flattening the confidence probabilities of all models
                    for i in range(len(p_1_s)):
                        p_1_s[i] = torch.ravel(p_1_s[i])
                        # print("p 1 s shape", p_1_s[i].shape)
                    
                    for i in range(len(p_1_s)):
                        al = (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) +  1e-7)).mean()
                        adv_loss.append(al)

                    adv_loss = sum(adv_loss)/len(adv_loss)
                # Adv loss calculation complete
                if adv_loss == []:
                    adv_loss = 0

                if alpha>1e-3:
                    loss = (1-alpha)*loss_ce + alpha*loss_kd + 0.1*adv_loss
                prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

                loss.backward()
                optimizer_s[i].step()

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
                
                if (iteration % args.print_freq == 0) or (iteration == len(train_loader)-1):
                    progress.display(iteration)
            
            
            scheduler_s[i].step(epoch)
            
            val_loss, val_acc1, val_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer_s[i], scheduler_s[i], network[i], valid_loader, criterion, args.eval_batch_size, mode='eval' )
            is_best = False 
            if val_acc1 > best_acc:
                best_acc = val_acc1
                is_best = True
                best_state_dict = copy.deepcopy( network[i].state_dict() )
                best_epoch = epoch+1
            save_checkpoint({
                        'epoch': epoch + 1,
                        'base_state_dict': base_model[i].state_dict(),
                        'best_acc': best_acc,
                        'scheduler_s' : scheduler_s[i].state_dict(),
                        'optimizer_s' : optimizer_s[i].state_dict(),
                    }, is_best, prefix=log_file_name)
                #val_losses.append(val_loss)
            logger.log('std: Valid eval after epoch: loss:{:.4f}\tlatest_acc:{:.2f}\tLR:{:.4f} -- best valacc {:.2f}'.format( val_loss,
                                                                                                                                val_acc1,
                                                                                                                                get_mlr(scheduler_s), 
                                                                                                                                best_acc))
        
            network[i].load_state_dict( best_state_dict )

    test_loss, test_acc1, test_acc5 = evaluate_model( network[i], test_loader, criterion, args.eval_batch_size )
    logger.log(
            "\n***{:s}*** [Post-train] [Student {}] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
                time_string(),i,
                test_loss,
                test_acc1,
                test_acc5,
                100 - test_acc1,
                100 - test_acc5,
            )
        )
    logger.log("Result is from best val model  of epoch:{}".format(best_epoch))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data Generation
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='./data/', help="The dataset name.")
    parser.add_argument("--model_name", type=str, default='ResNet32TwoPFiveM-NAS', help="The path to the model configuration")
    parser.add_argument("--teacher", type=str, default='ResNet10_l', help="teacher model name")
    parser.add_argument("--cutout_length", type=int, default=16, help="The cutout length, negative means not use.")
    parser.add_argument("--print_freq", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--print_freq_eval", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--save_dir", type=str, help="Folder to save checkpoints and log.", default='./kd_results/')
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers (default: 8)")
    parser.add_argument("--rand_seed1", type=int, help="base model seed")
    parser.add_argument("--rand_seed2", type=int, help="base model seed")
    parser.add_argument("--rand_seed3", type=int, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    #add_shared_args(parser)
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=200, help="Batch size for testing.")
    parser.add_argument('--epochs', type=int, default=100,help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0125,help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum')
    parser.add_argument("--loss_kd_frac", type=float, default=0.9, help="weight to the KD loss")
    parser.add_argument("--pretrained_student", type=bool, default=True, help="should I use CE-pretrained student?")
    parser.add_argument('--wd', type=float, default=0.00001,  help='weight decay')
    parser.add_argument('--temperature', type=int, default=4,  help='temperature for KD')
    parser.add_argument('--sched_cycles', type=int, default=1,  help='How many times cosine cycles for scheduler')

    parser.add_argument('--file_name', type=str, default="",  help='file_name')
    
    parser.add_argument('--k', type=int, default="1",  help='number_of_students')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 10)
    if (args.file_name is None or args.file_name == ""):
        args.file_name = "{}_{}_KD_seed-{}_cycles-{}_KDfrac-{}_T-{}_{}".format( args.model_name,args.dataset,
                                                            args.rand_seed, 
                                                            args.sched_cycles,
                                                            args.loss_kd_frac,
                                                            args.temperature,
                                                            args.epochs)
    assert args.save_dir is not None, "save-path argument can not be None"
    torch.manual_seed(args.rand_seed)
    main(args)




