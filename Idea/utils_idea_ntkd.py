import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import sys

from tqdm import tqdm
sys.path.append('../../')
from Idea.params import args_parser
from Idea.DKD import _get_gt_mask, _get_other_mask, dkd_loss

args = args_parser()
Scenario = args.Scenario
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
device_ids = args.device_ids
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        total = 0
        top1 = 0
        top5 = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)

            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)

        top1acc= round(100 * top1 / total,2)
        top5acc= round(100 * top5 / total,2)
        logger.info('Accuracy of the network on total {} test images: @top1={}%; @top5={}%'.
              format(total,top1acc,top5acc))
    if Scenario =='Digits':
        return top1acc
    else:
        return top5acc

def evaluate_network_generalization(network,dataloader_list,particiapnt_index,logger):
    generalization_list = []
    network.eval()
    with torch.no_grad():
        for index,dataloader in enumerate(dataloader_list):
            if index != particiapnt_index:
                total = 0
                top1 = 0
                top5 = 0
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = network(images)

                    _, max5 = torch.topk(outputs, 5, dim=-1)
                    labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

                    top1 += (labels == max5[:, 0:1]).sum().item()
                    top5 += (labels == max5).sum().item()
                    total += labels.size(0)

                top1acc = round(100 * top1 / total, 3)
                top5acc = round(100 * top5 / total, 3)
                if Scenario =='Digits':
                    generalization_list.append(top1acc)
                else:
                    generalization_list.append(top5acc)
            else:
                generalization_list.append(0)
    return generalization_list


def update_model_via_private_data(network,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    criterion = nn.CrossEntropyLoss()
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), loss.item()))
    return network,participant_local_loss_batch_list

def prox(network,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    criterion = nn.CrossEntropyLoss()
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), loss.item()))
    return network,participant_local_loss_batch_list


def update_model_via_private_data_with_support_model(network,frozen_network,temperature,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    if loss_function =='CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='KLDivLoss':
        criterion = nn.KLDivLoss(reduction='batchmean')
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            with torch.no_grad():
                soft_labels = F.softmax(frozen_network(images)/temperature,dim=1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            logsoft_outputs = F.log_softmax(outputs/temperature,dim=1)
            loss_soft = criterion(logsoft_outputs, soft_labels)
            loss_hard = criterion_hard(outputs, labels)
            loss = loss_hard+loss_soft
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_soft: {:.6f} Loss_hard: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), loss_soft.item(),loss_hard.item()))
    return network,participant_local_loss_batch_list

def update_model_via_private_data_with_two_model(network,frozen_network,progressive_network,temperature,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    if loss_function =='CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='KLDivLoss':
        criterion = nn.KLDivLoss(reduction='batchmean')
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate,weight_decay=1e-5)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            gt_mask = _get_gt_mask(outputs, labels)
            other_mask = _get_other_mask(outputs, labels)
            # 先进行sofamax
            softmax_outputs = F.softmax(outputs/temperature,dim=1)
            # 去除目标
            no_target_softmax_outputs = torch.where(gt_mask == 1,1,softmax_outputs)
            

            with torch.no_grad():
                progressive_output = progressive_network(images)
                # 先进行sofamax
                softmax_progressive_output = F.softmax(progressive_output/temperature,dim=1)
                # 去除目标
                no_target_softmax_progressive_outputs = torch.where(gt_mask == 1,1,softmax_progressive_output)
                
            
            
            inter_loss = criterion(torch.log(no_target_softmax_outputs),no_target_softmax_progressive_outputs)
            inter_loss = inter_loss*(temperature**2)
            # # 以下是直接拿dkd得
            # nckd_loss = dkd_loss(outputs, progressive_output, labels, 0, 1, temperature)
            loss_hard = criterion_hard(outputs, labels)

            loss = loss_hard+inter_loss
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\t nckd_loss: {:.6f} Loss_hard: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), inter_loss.item(),loss_hard.item()))
    return network,participant_local_loss_batch_list

def _train_net(index,net,inter_net,train_loader,local_lr,logger):
        participant_local_loss_batch_list = []
        #T = self.args.local_dis_power
        T = 3
        local_epoch = args.Private_Training_Epoch[0] # 50
        net = net.to(device)
        inter_net = inter_net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=local_lr, weight_decay=1e-5)
        criterionCE = nn.CrossEntropyLoss()
        criterionCE.to(device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(device)
        iterator = tqdm(range(local_epoch))
        for epoch_index in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                bs,class_num = outputs.shape
                soft_outputs = F.softmax(outputs/T,dim=1)
                non_targets_mask = torch.ones(bs,class_num).to(device).scatter_(1, labels.view(-1,1), 0)
                non_target_soft_outputs = soft_outputs[non_targets_mask.bool()].view(bs, class_num-1)

                non_target_logsoft_outputs = torch.log(non_target_soft_outputs)
                with torch.no_grad():
                    inter_outputs = inter_net(images)
                    soft_inter_outpus = F.softmax(inter_outputs/T,dim=1)
                    non_target_soft_inter_outputs = soft_inter_outpus[non_targets_mask.bool()].view(bs, class_num-1)
                inter_loss = criterionKL(non_target_logsoft_outputs, non_target_soft_inter_outputs)
                loss_hard = criterionCE(outputs, labels)
                inter_loss = inter_loss*(T**2)
                loss = loss_hard+ inter_loss
                optimizer.zero_grad()
                participant_local_loss_batch_list.append(loss.item())
                loss.backward()
                iterator.desc = "Local Pariticipant %d lossCE = %0.3f lossKD = %0.3f" % (index,loss_hard.item(),inter_loss.item())
                optimizer.step()
                if epoch_index % 5 ==0:
                    logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\t nckd_loss: {:.6f} Loss_hard: {:.6f}'.format(
                        epoch_index, batch_idx * len(images), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), inter_loss.item(),loss_hard.item()))
        return net,participant_local_loss_batch_list

