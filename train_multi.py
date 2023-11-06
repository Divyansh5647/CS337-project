import os, sys, time, torch, random, argparse, json, pickle, torch, copy
import itertools
from collections import namedtuple
import numpy as np
import pandas as pd
import datetime, pytz
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from torch.distributions import Categorical
from torch.utils.data import Dataset, random_split
from typing import Type, Any, Callable, Union, List, Optional
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
from copy import deepcopy
from pathlib import Path
from new_models import FullyConnectedNetwork
import matplotlib.pyplot as plt

from model_dict import get_model_from_name
from utils import get_model_infos
from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time
from starts import prepare_logger, prepare_seed
from get_dataset_with_transform import get_datasets
import torch.utils.data as data
from meta import *
from models import *
from utils import *
from DiSK import obtain_accuracy, get_mlr, save_checkpoint, evaluate_model
# from torchsummary import summary

def m__get_prefix( args ):
    prefix = args.file_name + '_' + args.dataset + '-' 
    return prefix

def get_model_prefix( args ):
    prefix = os.path.join(args.save_dir, m__get_prefix( args ) )
    return prefix

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

        loss = torch.mean(criterion(logits, targets))
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                progress.display(i)

    return losses.avg, top1.avg, top5.avg


def main(args):
    
    mul = 1
    Mode= 2 
    
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    logger = prepare_logger(args)
    prepare_seed(args.rand_seed)

    criterion_indiv = nn.CrossEntropyLoss(reduction= 'none')
    criterion = nn.CrossEntropyLoss()
    # LOADING DATA.........

    if args.dataset=="cifar100" or args.dataset=="cifar10": 
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
    
    """ 
    train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//5,  len(train_data)//5])
    test_data, valid_data = data.random_split(valid_data, [len(valid_data)-len(valid_data)//2,  len(valid_data)//2])
    """
    def _init_fn(worker_id):
        np.random.seed(int(args.rand_seed))
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    meta_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    valid_loader = torch.utils.data.DataLoader( # same data used for training metanet
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_init_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    meta_dataloader_iter = iter(train_loader)
    meta_dataloader_s_iter = iter(valid_loader)
  
    
    args.class_num = class_num
    
    logger.log(args.__str__())
    logger.log("Running in mode {}".format(Mode))
    logger.log("Train:{}\t, Valid:{}\t, Test:{}\n".format(len(train_data),
                                                          len(valid_data),
                                                          len(test_data)))
    logger.log("-" * 50)
    Arguments = namedtuple("Configure", ('class_num','dataset','dropout')  )
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset ,'dropout':0}
    model_config = Arguments(**md_dict)


    # Multiple students
    model_name = [ "ResNet10_xxxs","ResNet10_xxs","ResNet10_xs","ResNet10_s","ResNet10_m"]
    k= len(model_name)
    # Mode to change the input to the metanet - 1 to add features + input, 2- to add inptdirectly to the metanet
    image_encoder = ImageEncoder()
    image_encoder=image_encoder.cuda()
    
    base_model= [i for i in range(k)]
    network= [i for i in range(k)]
    best_state_dict= [i for i in range(k)]
    print("Number of Students = ",k)
    for i in range(k):
        base_model[i] = get_model_from_name( model_config, model_name[i] )
        logger.log("Student {} + {}:".format(i, model_name[i]) )
        if args.dataset == 'cifar100':
           ce_ptrained_path = "./ce_results/CE_with_seed-{}_cycles-1-_{}_{}-{}_"\
                            "model_best.pth.tar".format(args.rand_seed,
                                                        # args.sched_cycles,
                                                        model_name[i],
                                                        args.dataset,
                                                        model_name[i])
        else:
           ce_ptrained_path = "./ce_results_tinyImagenet/CE_with_seed-{}_cycles-1-_{}_{}-{}_"\
                            "model_best.pth.tar".format(args.rand_seed,
                                                        # args.sched_cycles,
                                                        model_name[i],
                                                        args.dataset,
                                                        model_name[i])
      
        # Loading multiple students from pretrained student......  
        logger.log("using pretrained student model from {}".format(ce_ptrained_path))
                        
        if args.pretrained_student: # load CE-pretrained student
            assert Path().exists(), "Cannot find the initialization file : {:}".format(ce_ptrained_path)
        base_checkpoint = torch.load(ce_ptrained_path)
        base_model[i].load_state_dict(base_checkpoint["base_state_dict"])
            
    
    for i in range(k):
        base_model[i] = base_model[i].cuda()
        network[i] = base_model[i] 
        
        best_state_dict[i] = copy.deepcopy( base_model[i].state_dict() )
        
      
    #testing pretrained student
    for i in range(k):
        test_loss, test_acc1, test_acc5 = evaluate_model( network[i], test_loader, criterion, args.eval_batch_size )
        logger.log(
        "***{:s}*** before training [Student(CE)] {} Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),i,
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
        )
    
    # lr = args.lr
    optimizer_s=[i for i in range(k)]
    scheduler_s=[i for i in range(k)]
    
    for i in range(k):
        optimizer_s[i] = torch.optim.SGD(base_model[i].parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler_s[i] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_s[i], args.epochs//args.sched_cycles)
        logger.log("Scheduling LR update to student no {}, {} time at {}-epoch intervals".format(k,args.sched_cycles, 
                                                                                        args.epochs//args.sched_cycles))

    # TEACHER
    Teacher_model = get_model_from_name( model_config, args.teacher )
    model_name_t = args.teacher
    # teach_PATH="/home/shashank/disk/model_10l/disk-CE-cifar100-ResNet10_l-model_best.pth.tar"
    if args.dataset == 'cifar100':
        teach_PATH = "./ce_results/CE_with_seed-{}_cycles-1-_{}_{}-{}_"\
                            "model_best.pth.tar".format(args.rand_seed,
                                                        # args.sched_cycles,
                                                        args.teacher,
                                                        args.dataset,
                                                        args.teacher)
    else:
        teach_PATH = "./ce_results_tinyImagenet/CE_with_seed-{}_cycles-1-_{}_{}-{}_"\
                    "model_best.pth.tar".format(args.rand_seed,
                                                        # args.sched_cycles,
                                                        args.teacher,
                                                        args.dataset,
                                                        args.teacher)
                    
                    
    teach_checkpoint = torch.load(teach_PATH)
    Teacher_model.load_state_dict(teach_checkpoint['base_state_dict'])
    Teacher_model = Teacher_model.cuda()
    network_t = Teacher_model
    network_t.eval()
    logger.log("Teacher loaded....")
    
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

    flop, param = get_model_infos(base_model[0], xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(base_model[0].get_message()))
    logger.log("-" * 50)
    logger.log("[Student]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3))

    
    # METANET
    if not args.inst_based: # inst_based is True, inst_based checks if our metanet will get instance-wise
        meta_net = MLP(hidden_size=args.meta_net_hidden_size,num_layers=args.meta_net_num_layers).to(device=args.device)
    elif args.meta_type == 'meta_lite':
        meta_net = InstanceMetaNetLite(num_layers=1).cuda()
        # meta_net = copy.deepcopy(network)
    elif args.meta_type == 'instance':
        logger.log("Using Instance metanet....")
        meta_net = InstanceMetaNet(input_size=args.input_size).cuda()
    elif args.single:
        logger.log("Using ResNet32 metanet for single....")
        if Mode == 6:
            logger.log("Using Mode 6")
            meta_net = ResNet32MetaNet(k=k,single=3).cuda()
        elif Mode == 7:
            meta_net = ResNet32MetaNet(k=k,single=4).cuda()
        else:
            logger.log("Using Mode default (k=k, single = 2)")
            meta_net = ResNet32MetaNet(k=k,single=2).cuda()
    elif args.correlated:
        logger.log("Using ResNet32 metanet for correlated....")
        if Mode == 6:
            meta_net = ResNet32MetaNet(k=k,single=7).cuda()
        elif Mode == 7:
            meta_net = ResNet32MetaNet(k=k,single=8).cuda()
        else:
            meta_net = ResNet32MetaNet(k=k,single=6).cuda()
    else:
        if Mode==1:
            logger.log("Using Fully Connected Network as MetaNet")
            meta_net = FullyConnectedNetwork(912,[128,256,517,1024,128], 2*k ).cuda()
            # meta_net = FullyConnectedNetwork(516,[128,256,517,1024,500,200,100,50,25,10], 2*k ).cuda()
        elif Mode==2 or Mode == 6:
            logger.log("Using ResNet32 metanet....")
            meta_net = ResNet32MetaNet(k=k).cuda()
            # meta_net_name = "ResNet34"
            # Arguments_meta = namedtuple("Configure", ('class_num','dataset','dropout')  )
            # md_dict_meta = { 'class_num' : 2*k, 'dataset' : args.dataset, 'dropout' : 0 }
            # model_config_meta = Arguments_meta(**md_dict_meta)
            # meta_net= get_model_from_name( model_config_meta, meta_net_name )
            # meta_net= meta_net.cuda()
            # logger.log("Loaded {} Metanet..".format(meta_net_name))
        elif Mode == 7:
            meta_net = ResNet32MetaNet(k=k,nout=4).cuda()
        elif Mode==3 or Mode==4:
            if Mode==3:
                logger.log("Using Fully Connected Network as MetaNet and logits+teacher")
            else:
                logger.log("Using Fully Connected Network as MetaNet and logits+one-hot")
            meta_net = FullyConnectedNetwork(class_num*(k+1),[128,256,517,1024,128], 2*k ).cuda()
        elif Mode==5:
            logger.log("Using Fully Connected Network as MetaNet and margin-vectors")
            meta_net = FullyConnectedNetwork(class_num*(k),[128,256,517,1024,128], 2*k ).cuda()

        # Print the summary of the metanet 
        print("MetaNet Summary")
        # summary(meta_net, (3,32,32), batch_size=400, device='cuda')
        
    
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    best_acc, best_epoch = 0.0, 0
    val_losses = []
    alphas_collection = [ [] for i in range(k)]
    betas_collection = [ [] for i in range(k)]
    if Mode == 6 or Mode ==7:
       gammas_collection = [ [] for i in range(k)]
       if Mode ==7:
           deltas_collection = [ [] for i in range(k)]
      
    Temp = args.temperature
    log_file_name = get_model_prefix( args )
    
    loss= [i for i in range(k)] 
    best_acc= [0 for i in range(k)]
    for epoch in range(args.epochs):
        logger.log("\nStarted EPOCH:{}".format(epoch))
        mode='train'
        logger.log('Training epoch', epoch)
        
        losses = [ AverageMeter('Loss', ':.4e') for i in range(k)]
        top1 = [ AverageMeter('Acc@1', ':6.2f') for i in range(k)]
        top5 = [ AverageMeter('Acc@5', ':6.2f') for i in range(k)]
        progress= [ i for i in range(k)]
        for i in range(k):
            base_model[i].train()
            progress[i] = ProgressMeter(
                    logger,
                    len(train_loader),
                    [losses[i], top1[i], top5[i]],
                    prefix="[{}] E: [{}]".format(mode.upper(), epoch))
        
        #alphas = [i for i in range(k)]
        #betas = [i for i in range(k)]
        #gammas = [i for i in range(k)]
        #deltas = [i for i in range(k)]

        for iteration, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = labels.cuda(non_blocking=True)
            #train metanet
            if (iteration) % args.meta_interval == 0:
                pseudo_net= [i for i in range(k)]
                features= [i for i in range(k)]
                pseudo_outputs= [i for i in range(k)]  
                 
                # make a descent in a COPY OF THE STUDENT (in train data), metanet net will do a move on this move for metaloss
                for i in range(k):
                    #print("Model name:", model_name[i])
                    pseudo_net[i] = get_model_from_name( model_config, model_name[i] )
                    pseudo_net[i] = pseudo_net[i].cuda()
                    pseudo_net[i].load_state_dict(network[i].state_dict()) # base_model == network
                    pseudo_net[i].train()
                    features[i], pseudo_outputs[i], _ = pseudo_net[i](inputs)
                    
                meta_net.train()    
                
                with torch.no_grad():                         
                    _, teacher_outputs, _ = network_t(inputs)

                if Mode==6:
                    with torch.no_grad():
                       pseudo_logits = pseudo_outputs.copy()
                    for i in range(k):
                        if i ==0:
                            logits_plus = pseudo_logits[i].detach().unsqueeze(2)
                        else:
                            logits_plus = torch.cat([ logits_plus,pseudo_logits[i].detach().unsqueeze(2)], dim=2)
                    
                    fake_teacher = torch.min(logits_plus,dim=2).values
                    max_values = torch.max(logits_plus,dim=2).values
                    fake_teacher[torch.arange(fake_teacher.size(0)),targets] = max_values[torch.arange(fake_teacher.size(0)),targets]

                pseudo_loss_vector_CE= [ i for i in range(k)]
                for i in range(k):
                    pseudo_loss_vector_CE[i]= criterion_indiv(pseudo_outputs[i], targets) # [B]
                    #pseudo_loss_vector_CE_reshape = torch.reshape(pseudo_loss_vector_CE, (-1, 1)) # [B, 1]
                
                if args.meta_type == 'meta_lite':
                    pseudo_hyperparams = meta_net(features)    
                else:
                    if Mode==2 or Mode==6 or Mode == 7:
                        # pseudo_hyperparams,_,_ = meta_net(inputs)
                        pseudo_hyperparams = meta_net(inputs)
                    elif Mode==1:
                        with torch.no_grad():
                            image_features = image_encoder(inputs)
                            tensors = logits.copy()
                            encoded_img= image_features
                            for i in range(k):
                                encoded_img= torch.cat([ encoded_img,tensors[i].view(-1, 1)], dim=1)
                            pseudo_hyperparams = meta_net(encoded_img)
                    elif Mode==3 or Mode==4:
                        
                        pseudo_logits = pseudo_outputs.copy()
                        if Mode==3:
                            logits_plus = teacher_outputs
                        else:
                           logits_plus = F.one_hot(targets,num_classes=class_num)
                        
                        for i in range(k):
                            logits_plus = torch.cat([ logits_plus,pseudo_logits[i]], dim=1)

                        pseudo_hyperparams = meta_net(logits_plus)
                    
                    elif Mode == 5:
                        pseudo_logits = pseudo_outputs.copy()

                        for i in range(k):
                            pseudo_logits_i = pseudo_logits[i]
                            lmargin = pseudo_logits_i[torch.arange(pseudo_logits_i.size(0)),targets][:,None] - pseudo_logits_i
                            if i ==0:
                                logits_plus = lmargin
                            else:
                                logits_plus = torch.cat([ logits_plus,lmargin], dim=1)
                        
                        pseudo_hyperparams = meta_net(logits_plus)
                            
                
                # print(pseudo_hyperparams.shape)                    
                
                Temp = args.temperature
                pseudo_loss_vector_KD= [ i for i in range(k)]
                if Mode==6 or Mode == 7:
                   pseudo_loss_vector_KD_fake= [ i for i in range(k)]
                for i in range(k):
                    pseudo_loss_vector_KD[i]= nn.KLDivLoss(reduction='none')(             # [B x n]
                                    F.log_softmax(pseudo_outputs[i] / Temp, dim=1),
                                    F.softmax(teacher_outputs / Temp, dim=1))
                    
                    if Mode==6:
                        pseudo_loss_vector_KD_fake[i]= nn.KLDivLoss(reduction='none')(             # [B x n]
                                    F.log_softmax(pseudo_outputs[i] / Temp, dim=1),
                                    F.softmax(fake_teacher / Temp, dim=1))
                
                # print(pseudo_hyperparams.shape)
                #for i in range(k):
                if args.single:
                    alpha = mul*pseudo_hyperparams[:,0]
                    beta = mul*pseudo_hyperparams[:,1]
                    if Mode==6 or Mode == 7:
                        gamma = mul*pseudo_hyperparams[:,2]
                        if Mode==7:
                            delta = mul*pseudo_hyperparams[:,3]
                elif args.correlated:
                    alpha = mul*pseudo_hyperparams[:,:k]
                    beta = mul*pseudo_hyperparams[:,1:(k+1)]
                    if Mode==6 or Mode == 7:
                        gamma = mul*pseudo_hyperparams[:,2:(k+2)]
                        if Mode==7:
                            delta = mul*pseudo_hyperparams[:,3:]
                else:
                    alpha = mul*pseudo_hyperparams[:,:k]
                    beta = mul*pseudo_hyperparams[:,k:2*k]
                    if Mode==6 or Mode == 7:
                        gamma = mul*pseudo_hyperparams[:,2*k:3*k]
                        if Mode==7:
                            delta = mul*pseudo_hyperparams[:,3*k:]

                if Mode==7:
                    with torch.no_grad():
                       pseudo_logits = pseudo_outputs.copy()
                    for i in range(k):
                        if i == 0:
                           fake_teacher = delta[:,i][:,None]*pseudo_logits[i].detach()
                        else:
                            fake_teacher += delta[:,i][:,None]*pseudo_logits[i].detach()
                    for i in range(k):                    
                        pseudo_loss_vector_KD_fake[i]= nn.KLDivLoss(reduction='none')(             # [B x n]
                                    F.log_softmax(pseudo_outputs[i] / Temp, dim=1),
                                    F.softmax(fake_teacher / Temp, dim=1))
                
                #LOSS Update....
                loss_CE=[i for i in range(k)]
                loss_KD=[i for i in range(k)]
                if Mode==6 or Mode==7:
                   loss_KD_fake=[i for i in range(k)]
                
                pseudo_loss= [i for i in range(k)]
                for i in range(k):
                    if args.single:
                        loss_CE[i] = torch.mean( alpha*pseudo_loss_vector_CE[i] )
                        loss_KD[i] = (Temp**2)* torch.mean( beta* torch.sum(pseudo_loss_vector_KD[i],dim=1))
                        if Mode==6 or Mode==7:
                            loss_KD_fake[i] = (Temp**2)* torch.mean( gamma* torch.sum(pseudo_loss_vector_KD_fake[i],dim=1))
                    else:
                        loss_CE[i] = torch.mean( alpha[:,i]*pseudo_loss_vector_CE[i] )
                        loss_KD[i] = (Temp**2)* torch.mean( beta[:,i]* torch.sum(pseudo_loss_vector_KD[i],dim=1))
                        if Mode==6 or Mode==7:
                            loss_KD_fake[i] = (Temp**2)* torch.mean( gamma[:,i]* torch.sum(pseudo_loss_vector_KD_fake[i],dim=1))
                    
                    if Mode==6 or Mode==7:
                        pseudo_loss[i] = loss_CE[i] + loss_KD[i] + loss_KD_fake[i]
                    else:
                        pseudo_loss[i] = loss_CE[i] + loss_KD[i]
                
                pseudo_grads=[i for i in range(k)]
                
                for i in range(k):
                    pseudo_grads[i] = torch.autograd.grad(pseudo_loss[i], pseudo_net[i].parameters(), create_graph=True)

                    # using the current student's LR to train pseudo
                    base_model_lr = optimizer_s[i].param_groups[0]['lr']
                    pseudo_optimizer = MetaSGD(pseudo_net[i], pseudo_net[i].parameters(), lr=base_model_lr)
                    pseudo_optimizer.load_state_dict(optimizer_s[i].state_dict())
                    pseudo_optimizer.meta_step(pseudo_grads[i])

                del pseudo_grads

                # NOW, do metanet descent
                # cycle through the metadata used for validation
                try:
                    valid_inputs, valid_labels = next(meta_dataloader_iter)
                except StopIteration:
                    print("Loading Meta training data")
                    meta_dataloader_iter = iter(meta_loader)
                    valid_inputs, valid_labels = next(meta_dataloader_iter)

                
                valid_inputs, valid_labels = valid_inputs.cuda(), valid_labels.cuda()
                meta_loss = 0
                for i in range(k):
                    _,meta_outputs,_ = pseudo_net[i](valid_inputs) # apply the stepped pseudo net on the validation data

                    meta_loss += torch.mean(criterion_indiv(meta_outputs, valid_labels.long())) + \
                               args.mcd_weight*mcd_loss(pseudo_net[i], valid_inputs)
                    #meta_loss += 2*(k-i)*(torch.mean(criterion_indiv(meta_outputs, valid_labels.long())) + \
                    #            args.mcd_weight*mcd_loss(pseudo_net[i], valid_inputs)) #i+1 #k-i
                
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()
            
            for i in range(k):
                optimizer_s[i].zero_grad() 

            features= [i for i in range(k)]
            logits= [i for i in range(k)]
            loss_vector= [i for i in range(k)]
            
            for i in range(k):
                features[i], logits[i], _ = network[i](inputs)
                loss_vector[i] = criterion_indiv(logits[i], targets)   
            with torch.no_grad():
                _,teacher_outputs , _ = network_t(inputs)

            if Mode==6:
                with torch.no_grad():
                    pseudo_logits = logits.copy()
                for i in range(k):
                    if i ==0:
                        logits_plus = pseudo_logits[i].detach().unsqueeze(2)
                    else:
                        #lreshape = pseudo_logits[i].detach().view(pseudo_logits[i].size(0),pseudo_logits[i].size(1),1)
                        logits_plus = torch.cat([ logits_plus,pseudo_logits[i].detach().unsqueeze(2)], dim=2)
                
                fake_teacher = torch.min(logits_plus,dim=2).values
                max_values = torch.max(logits_plus,dim=2).values
                fake_teacher[torch.arange(fake_teacher.size(0)),targets] = max_values[torch.arange(fake_teacher.size(0)),targets]

            with torch.no_grad():
                if args.meta_type == 'meta_lite':
                    hyperparams = meta_net(features) 
                    # hyperparams,_, _ = meta_net(features)     
                else:
                    if Mode==1:
                        image_features = image_encoder(inputs)
                        tensors = logits.copy()
                        #tensors =pseudo_loss_vector_CE.copy()
                        encoded_img= image_features
                        for i in range(k):
                            # APPENDING THE LOGITSOF THE STUDENTS
                            encoded_img= torch.cat([ encoded_img,tensors[i]], dim=1)
                            loss_vector[i] = criterion_indiv(logits[i], targets)
                        hyperparams = meta_net(encoded_img)
                    elif Mode==2 or Mode ==6 or Mode ==7:
                        # hyperparams,_,_ = meta_net(inputs)
                        hyperparams = meta_net(inputs)

                    elif Mode==3 or Mode==4:
                        
                        pseudo_logits = logits.copy()
                        if Mode==3:
                            logits_plus = teacher_outputs
                        else:
                           logits_plus = F.one_hot(targets,num_classes=class_num)
                        
                        for i in range(k):
                            logits_plus = torch.cat([logits_plus,pseudo_logits[i]], dim=1)

                        hyperparams = meta_net(logits_plus)
                    
                    elif Mode == 5:
                        pseudo_logits = logits.copy()

                        for i in range(k):
                            pseudo_logits_i = pseudo_logits[i]
                            lmargin = pseudo_logits_i[torch.arange(pseudo_logits_i.size(0)),targets][:,None] - pseudo_logits_i
                            if i ==0:
                                logits_plus = lmargin
                            else:
                                logits_plus = torch.cat([ logits_plus,lmargin], dim=1)
                        
                        hyperparams = meta_net(logits_plus)
            
            # print(pseudo_hyperparams.shape)
            #for i in range(k):
            if args.single:
                alpha = mul*hyperparams[:,0]
                beta = mul*hyperparams[:,1]
                if Mode==6 or Mode == 7:
                    gamma = mul*hyperparams[:,2]
                    if Mode==7:
                        delta = mul*hyperparams[:,3]
            elif args.correlated:
                alpha = mul*hyperparams[:,:k]
                beta = mul*hyperparams[:,1:(k+1)]
                if Mode==6 or Mode == 7:
                    gamma = mul*hyperparams[:,2:(k+2)]
                    if Mode==7:
                        delta = mul*hyperparams[:,3:]
            else:
                alpha = mul*hyperparams[:,:k]
                beta = mul*hyperparams[:,k:2*k]
                if Mode==6 or Mode == 7:
                    gamma = mul*hyperparams[:,2*k:3*k]
                    if Mode==7:
                        delta = mul*hyperparams[:,3*k:]

            if Mode==7:
                with torch.no_grad():
                    pseudo_logits = logits.copy()
                for i in range(k):
                    if i == 0:
                        fake_teacher = delta[:,i][:,None]*pseudo_logits[i].detach()
                    else:
                        fake_teacher += delta[:,i][:,None]*pseudo_logits[i].detach()
                #for i in range(k):                    
                #    pseudo_loss_vector_KD_fake[i]= nn.KLDivLoss(reduction='none')(             # [B x n]
                #                F.log_softmax(logits[i] / Temp, dim=1),
                #                F.softmax(fake_teacher / Temp, dim=1))
                

            #for j in range(k):
                
            if iteration == 0:
                alphas = alpha.detach().cpu()#.numpy().tolist()
                betas = beta.detach().cpu()
                if Mode==6 or Mode==7:
                    gammas = gamma.detach().cpu()
                    if Mode==7:
                        deltas = delta.detach().cpu()        
            else:
                alphas = torch.cat((alphas,alpha.detach().cpu()), dim =0)
                betas = torch.cat((betas,beta.detach().cpu()), dim =0)
                if Mode==6 or Mode==7:
                    gammas = torch.cat((gammas,gamma.detach().cpu()), dim =0)
                    if Mode==7:
                        deltas = torch.cat((deltas,delta.detach().cpu()), dim =0)

            for i in range(k):
                pseudo_loss_vector_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(logits[i] / Temp, dim=1),\
                                                                F.softmax(teacher_outputs / Temp, dim=1))
                
                if Mode==6 or Mode==7:
                    pseudo_loss_vector_KD_fake = nn.KLDivLoss(reduction='none')(F.log_softmax(logits[i] / Temp, dim=1),\
                                                                F.softmax(fake_teacher / Temp, dim=1))
                
                #LOSS Update...
                if args.single:
                    loss_CE = torch.mean( alpha*loss_vector[i])
                    loss_KD = (Temp**2)* torch.mean(beta* torch.sum(pseudo_loss_vector_KD, dim=1))
                    if Mode==6 or Mode==7:
                        loss_KD_fake = (Temp**2)* torch.mean( gamma*torch.sum(pseudo_loss_vector_KD_fake,dim=1))
                else:
                    loss_CE = torch.mean( alpha[:,i]*loss_vector[i])
                    loss_KD = (Temp**2)* torch.mean(beta[:,i]* torch.sum(pseudo_loss_vector_KD, dim=1))
                    if Mode==6 or Mode==7:
                        loss_KD_fake = (Temp**2)* torch.mean( gamma[:,i]*torch.sum(pseudo_loss_vector_KD_fake,dim=1))

                if Mode==6 or Mode==7:
                    loss = loss_CE + loss_KD + loss_KD_fake
                else:
                    loss = loss_CE + loss_KD
                prec1, prec5 = obtain_accuracy(logits[i].data, targets.data, topk=(1, 5))

                loss.backward()
                optimizer_s[i].step()

                losses[i].update(loss.item(), inputs.size(0))
                top1[i].update(prec1.item(), inputs.size(0))
                top5[i].update(prec5.item(), inputs.size(0))
                # scheduler_s.step(epoch+iteration/len(train_loader))
            
            for i in range(k):
                if (iteration % args.print_freq == 0) or (iteration == len(train_loader)-1):
                    print("Student:",model_name[i])
                    progress[i].display(iteration)
        
        if epoch%100==0 or epoch==args.epochs-1:
            #log_alphas_collection.append(torch.log(alphas))
            #log_betas_collection.append(torch.log(betas))
            #for i in range(k):
            alphas_collection.append(alphas)
            betas_collection.append(betas)
            if Mode == 6 or Mode ==7:
                gammas_collection.append(gammas)
                if Mode == 7:
                    deltas_collection.append(deltas)
        
        '''
        logger.log("alpha quartiles: \nq0\tq25\tq50\tq75\tq100\n{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f} with std {:.6f}".format(
                                                                    torch.quantile(alphas, 0.0),
                                                                    torch.quantile(alphas, 0.25),
                                                                    torch.quantile(alphas, 0.5),
                                                                    torch.quantile(alphas, 0.75),
                                                                    torch.quantile(alphas, 1),
                                                                    torch.std(alphas))) 
        logger.log("beta quartiles: \nq0\tq25\tq50\tq75\tq100\n{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f} with std {:.6f}".format(
                                                                        torch.quantile(betas, 0.0),
                                                                        torch.quantile(betas, 0.25),
                                                                        torch.quantile(betas, 0.5),
                                                                        torch.quantile(betas, 0.75),
                                                                        torch.quantile(betas, 1),
                                                                        torch.std(betas)))   
        '''
        for i in range(k):
            scheduler_s[i].step(epoch)

        save_checkpoint({
                'meta_state_dict': meta_net.state_dict(),
                'optimizer_m' : meta_optimizer.state_dict(),
            }, False, prefix=log_file_name+ "metanet")
        for i in range(k):
            val_loss, val_acc1, val_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer_s[i], scheduler_s[i], network[i], valid_loader, criterion, args.eval_batch_size, mode='eval' )
            is_best = False 
            if val_acc1 > best_acc[i]:
                best_acc[i] = val_acc1
                is_best = True
                best_state_dict[i] = copy.deepcopy( network[i].state_dict() )
                best_epoch = epoch+1
            save_checkpoint({
                    'epoch': epoch + 1,
                    'student': i,
                    'base_state_dict': base_model[i].state_dict(),
                    'best_acc': best_acc[i],
                    'scheduler_s' : scheduler_s[i].state_dict(),
                    'optimizer_s' : optimizer_s[i].state_dict(),
                }, is_best, prefix=log_file_name+ model_name[i])
            val_losses.append(val_loss)
            logger.log('std {} Valid eval after epoch: loss:{:.4f}\tlatest_acc:{:.2f}\tLR:{:.4f} -- best valacc {:.2f}'.format( i,val_loss,
                                                                                                                            val_acc1,
                                                                                                                            get_mlr(scheduler_s[i]), 
                                                                                                                            best_acc[i]))

    for i in range(k):
        network[i].load_state_dict( best_state_dict[i] )
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
        logger.log("Result is from best val model {} of epoch:{}".format(i,best_epoch))
    
    plots_dir = os.path.join(args.save_dir, args.file_name)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    """
    # post valid loss
    fig, ax = plt.subplots()
    ax.plot(val_losses) 
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')   
    fig.savefig(os.path.join(plots_dir, 'valid_loss.png'))
    """
    #for i in range(k):
    with open(os.path.join(plots_dir, 'alpha.pkl'), 'wb') as f:
        pickle.dump(alphas_collection, f)
        logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'alpha_dump.pkl')))
    with open(os.path.join(plots_dir, 'beta.pkl'), 'wb') as f:
        pickle.dump(betas_collection, f)
        logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'beta_dump.pkl'))) 
    if Mode == 6 or Mode == 7:
        with open(os.path.join(plots_dir, 'gamma.pkl'), 'wb') as f:
            pickle.dump(gammas_collection, f)
            logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'gamma_dump.pkl')))
        if Mode == 7:
            with open(os.path.join(plots_dir, 'delta.pkl'), 'wb') as f:
                pickle.dump(deltas_collection, f)
                logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'delta_dump.pkl')))
    

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='./data/', help="The dataset name.")
    parser.add_argument("--model_name", type=str, default='ResNet32TwoPFiveM-NAS', help="The path to the model configuration")
    parser.add_argument("--teacher", type=str, default='ResNet10_l', help="teacher model name")
    parser.add_argument("--cutout_length", type=int, default=16, help="The cutout length, negative means not use.")
    parser.add_argument("--print_freq", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--print_freq_eval", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--save_dir", type=str, help="Folder to save checkpoints and log.", default='./logs/')
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers (default: 8)")
    parser.add_argument("--rand_seed", type=int, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    #add_shared_args(parser)
    parser.add_argument("--batch_size", type=int, default=400, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=400, help="Batch size for testing.")
    parser.add_argument('--epochs', type=int, default=100,help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=10e-4   ,help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum')
    parser.add_argument("--pretrained_student", type=int, default=1, help="should I use CE-pretrained student?")
    parser.add_argument('--wd', type=float, default=0.0005,  help='weight decay')
    parser.add_argument('--label', type=str, default="",  help='give some label you want appended to log fil')
    parser.add_argument('--temperature', type=int, default=4,  help='temperature for KD')
    parser.add_argument('--sched_cycles', type=int, default=1,  help='How many times cosine cycles for scheduler')

    parser.add_argument('--file_name', type=str, default="",  help='file_name')
    #parser.add_argument('--type', type=int, default="1",  help='if 1, using normal meta net, other wise custom(ip is op of stds)')
    
    #####################################################################
    
    #parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--inst_based', type=bool, default=True)
    parser.add_argument('--meta_interval', type=int, default=20)
    parser.add_argument('--mcd_weight', type=float, default=0.5)
    parser.add_argument('--meta_weight_decay', type=float, default=1e-4)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--unsup_adapt', type=bool, default=False)
    parser.add_argument('--meta_type', type=str, default='resnet') # or meta_lite or instance
    parser.add_argument('--single', type=bool, default=False)
    parser.add_argument('--correlated', type=bool, default=False)
    #####################################################################
    
    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 10)
    if (args.file_name is None or args.file_name == ""):
        if args.pretrained_student==1:
            args.file_name = "{}_multi_net_{}_meta_seed-{}".format(args.teacher, args.dataset,
                                                            args.rand_seed, 
                                                            )
        else:
            args.file_name = "meta_no_PT_seed-{}_metalr-{}_T-{}_{}_{}-cycles".format(
                                                            args.rand_seed, 
                                                            args.meta_lr,
                                                            args.temperature,
                                                            args.epochs,
                                                            args.sched_cycles)
    args.file_name += '_'+args.meta_type
    assert args.save_dir is not None, "save-path argument can not be None"
    os.makedirs(args.save_dir,exist_ok=True)
    main(args)
