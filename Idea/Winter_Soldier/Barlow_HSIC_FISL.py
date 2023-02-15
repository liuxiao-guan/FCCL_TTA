import sys
import os
sys.path.append('../../')
sys.path.append('.')
from Network.utils_network import init_nets
from Dataset.utils_dataset import init_logs, get_dataloader,generate_public_data_idxs
from Idea.utils_idea_dkd import update_model_via_private_data_with_two_model,evaluate_network,mkdirs
from Idea.params import args_parser
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import *
import numpy as np
import torch
import copy
import os
import torch

args = args_parser()
# if args.deterministic:
#     cudnn.benchmark = False # 程序在开始时花时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法 卷积神经网络（比如 ResNet-101）
#     cudnn.deterministic = True #每次卷积都会选用相同的算法
#     random.seed(args.Seed)
#     np.random.seed(args.Seed)
#     torch.manual_seed(args.Seed)
#     torch.cuda.manual_seed(args.Seed)
'''
Global Parameters
'''
Method_Name = 'Winter_Soldier'
Ablation_Name='Barlow_HSIC_FISL'

Temperature = 1
Ntkd_Temperature = 3
Scenario = args.Scenario
Seed = args.Seed
N_Participants = args.N_Participants
CommunicationEpoch = args.CommunicationEpoch
Local_TrainBatchSize = args.Local_TrainBatchSize
TrainBatchSize = args.TrainBatchSize
TestBatchSize = args.TestBatchSize
Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Idea_Winter_Dir = args.Project_Dir+'Idea/Winter_Soldier/'
Private_Net_Name_List = args.Private_Net_Name_List
Private_Net_Feature_Dim_List = args.Private_Net_Feature_Dim_List
Pariticpant_Params = {
    'loss_funnction' : 'KLDivLoss',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
'''
Scenario for large domain gap
'''
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Data_Total_Len_List = args.Private_Data_Total_Len_List
Private_Data_Len_List = args.Private_Data_Len_List
Private_Training_Epoch = args.Private_Training_Epoch
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)
'''
Public data parameters
'''
Public_Dataset_Name = args.Public_Dataset_Name
Public_Dataset_Length = args.Public_Dataset_Length
Public_Dataset_Dir = Dataset_Dir+Public_Dataset_Name
Public_Training_Epoch = args.Public_Training_Epoch

# 获取非对角线的值
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
 
# 进行Instance-wise Similarity 使h1-->s1
def instance_wise(linear_output):
    # l2归一化 方法
    # l2 范数 

    h_12 = linear_output.pow(2).sum(1, keepdim=True).pow(1. / 2) ## 512*1
    # 矩阵除以 l2 范数
    h_1 = linear_output.div(h_12)
    # 得到s_1s_1
    s_1 = h_1 @ h_1.T/(0.02) 
    # 去掉对角线位置
    diag = torch.diag(s_1) ## 全是50
    a_diag = torch.diag_embed(diag)
    s_1 = s_1 - a_diag
    return s_1

def _calculate_isd_sim(features):
    sim_q = torch.mm(features,features.T)
    logits_mask = torch.scatter(
        torch.ones_like(sim_q),
        1,
        torch.arange(sim_q.size(0)).view(-1, 1).to(device),
        0
    )
    row_size = sim_q.size(0)
    sim_q = sim_q[logits_mask.bool()].view(row_size,-1)
    return sim_q/0.02

if __name__ =='__main__':
    logger = init_logs(sub_name=Ablation_Name)
    logger.info('Method Name : '+Method_Name + ' Ablation Name : '+Ablation_Name)
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    device_ids = args.device_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
   
    logger.info("Initialize Participants' Data idxs and Model")
    # For Digits scenario
    private_dataset_idxs_dict = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(Private_Data_Total_Len_List[index])
        idxes = idxes[0:Private_Data_Len_List[index]]
        private_dataset_idxs_dict[Private_Dataset_Name_List[index]]= idxes
    logger.info(private_dataset_idxs_dict)


    net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    logger.info("Load Participants' Models")
    for i in range(N_Participants):
        network = net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))

    frozen_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    logger.info("Load Frozen Participants' Models")
    for i in range(N_Participants):
        network = frozen_net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))

    progressive_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    logger.info("Load Progressive Participants' Models")
    for i in range(N_Participants):
        network = progressive_net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))

    logger.info("Initialize Public Data Parameters")
    print(Scenario+Public_Dataset_Name)
    public_data_indexs = generate_public_data_idxs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length)

    public_train_dl, _, _, _ = get_dataloader(dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir,
                                                            train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                            dataidxs=public_data_indexs)
    logger.info('Initialize Private Data Loader')
    private_train_data_loader_list = []
    private_test_data_loader_list = []
    for participant_index in range(N_Participants):
        private_dataset_name = Private_Dataset_Name_List[participant_index]
        private_dataidx = private_dataset_idxs_dict[private_dataset_name]
        private_dataset_dir = Dataset_Dir + private_dataset_name
        train_dl_local, test_dl_local, _, _ = get_dataloader(dataset=private_dataset_name,
                                                 datadir=private_dataset_dir,
                                                 train_bs=Local_TrainBatchSize, test_bs=TestBatchSize,
                                                 dataidxs=private_dataidx)
        private_train_data_loader_list.append(train_dl_local)
        private_test_data_loader_list.append(test_dl_local)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    # 开始训练
    for epoch_index in range(CommunicationEpoch):

        logger.info("The "+str(epoch_index)+" th Communication Epoch")
        logger.info('Evaluate Models')
        acc_epoch_list = []
        # 每一个epoch前进行模型的分析
        for participant_index in range(N_Participants):
            netname = Private_Net_Name_List[participant_index]
            private_dataset_name = Private_Dataset_Name_List[participant_index]
            private_dataset_dir = Dataset_Dir + private_dataset_name
            print(netname + '_' + private_dataset_name + '_' + private_dataset_dir)
            _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                              train_bs=TrainBatchSize,
                                              test_bs=TestBatchSize, dataidxs=None)
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
        acc_list.append(acc_epoch_list)

        a = datetime.now()
        # 开始全局训练
        '''
        Update Participants' Models via Public Data
        '''
        epoch_loss_dict = {}
        for pub_epoch_idx in range(Public_Training_Epoch):
            for batch_idx, (images, _) in enumerate(public_train_dl):
                batch_loss_dict  = {}
                col_loss_batch_list = []
                linear_output_list = []
                linear_output_target_list = []
                logitis_sim_list = []
                logits_sim_target_list = []
                images = images.to(device)

                for participant_index in range(N_Participants):

                    net = net_list[participant_index]
                    net = nn.DataParallel(net, device_ids=device_ids).to(device)
                    net.train()
                    linear_output  = net(images)
                    linear_output_target_list.append(linear_output.clone().detach())
                    linear_output_list.append(linear_output)

                    # # 特征提取 拷贝网络
                    # linear_feature_extractor = copy.deepcopy(net)
                    # Private_Net_Feature_Dim = Private_Net_Feature_Dim_List[participant_index]
                    # # 将网络的最后一层线性输入输出维度改成一样 其值就是特征提取器输出的维度d
                    # linear_feature_extractor.linear = nn.Linear(Private_Net_Feature_Dim, Private_Net_Feature_Dim)
	                # # ---以下几行必须要有：---
                    # torch.nn.init.eye_(linear_feature_extractor.linear.weight)
                    # torch.nn.init.zeros_(linear_feature_extractor.linear.bias)
                    # for param in linear_feature_extractor.parameters():
                    #     param.requires_grad = False
                    # features = linear_feature_extractor(x=images)

                    features = net.module.features(images)
                    features = F.normalize(features, dim=1)
                    logits_sim = _calculate_isd_sim(features)
                    logits_sim_target_list.append(logits_sim.clone().detach())
                    logitis_sim_list.append(logits_sim)


                for participant_index in range(N_Participants):
                    net = net_list[participant_index]
                    net = nn.DataParallel(net, device_ids=device_ids).to(device)
                    '''
                    FCCL Loss for overall Network
                    '''
                    optimizer = optim.Adam(net.parameters(), lr=Pariticpant_Params['learning_rate'])

                    linear_output = linear_output_list[participant_index]
                    linear_output_target_avg_list = []
                    for k in range(N_Participants):
                        linear_output_target_avg_list.append(linear_output_target_list[k])
                    linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)

                    z_1_bn = (linear_output-linear_output.mean(0))/linear_output.std(0)
                    z_2_bn = (linear_output_target_avg-linear_output_target_avg.mean(0))/linear_output_target_avg.std(0)
                    c = z_1_bn.T @ z_2_bn
                    c.div_(len(images))
                    # if batch_idx == len(publoader) - 3:
                    #     c_array = c.detach().cpu().numpy()
                    #     self._draw_heatmap(c_array, self.NAME, communication_idx, net_idx)

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                    fccl_loss = on_diag + 0.0051 * off_diag

                    logits_sim = logitis_sim_list[participant_index]
                    logits_sim_target_avg_list = []
                    for k in range(N_Participants):
                        logits_sim_target_avg_list.append(logits_sim_target_list[k])
                    logits_sim_target_avg = torch.mean(torch.stack(logits_sim_target_avg_list), 0)

                    inputs = F.log_softmax(logits_sim, dim=1)
                    targets = F.softmax(logits_sim_target_avg, dim=1)
                    loss_distill = F.kl_div(inputs, targets, reduction='batchmean')
                    loss_distill =3 * loss_distill+fccl_loss

                    optimizer.zero_grad()
                    col_loss = loss_distill
                    batch_loss_dict[participant_index]={'fccl':round(fccl_loss.item(),3),'distill':round(loss_distill.item(),3)}
                    col_loss_batch_list.append(col_loss.item())
                    if batch_idx == len(public_train_dl)-2:
                        print('Communcation: '+str(epoch_index)+'Net: '+str(participant_index)+'FCCL: '+str(round(fccl_loss.item(),3))+'Disti: '+str(round(loss_distill.item(),3)))
                    col_loss.backward()
                    optimizer.step()
                epoch_loss_dict[batch_idx]=batch_loss_dict
                col_loss_list.append(col_loss_batch_list)


        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()

            frozen_network = frozen_net_list[participant_index]
            frozen_network = nn.DataParallel(frozen_network,device_ids=device_ids).to(device)
            frozen_network.eval()

            progressive_network = progressive_net_list[participant_index]
            progressive_network = nn.DataParallel(progressive_network,device_ids=device_ids).to(device)
            progressive_network.eval()

            private_dataset_name=  Private_Dataset_Name_List[participant_index]
            private_dataidx = private_dataset_idxs_dict[private_dataset_name]
            private_dataset_dir = Dataset_Dir+private_dataset_name
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=private_dataset_name,
                                                                  datadir=private_dataset_dir,
                                                                  train_bs=Local_TrainBatchSize, test_bs=TestBatchSize,
                                                                  dataidxs=private_dataidx)

            private_epoch = max(int(Public_Dataset_Length / len(train_ds_local)), 1)
            private_epoch = Private_Training_Epoch[participant_index]

            network, private_loss_batch_list = update_model_via_private_data_with_two_model(network=network,
            frozen_network=frozen_network,progressive_network=progressive_network,
            temperature = Ntkd_Temperature,private_epoch=private_epoch,private_dataloader=train_dl_local,
            loss_function=Pariticpant_Params['loss_funnction'],optimizer_method=Pariticpant_Params['optimizer_name'],
            learing_rate=Pariticpant_Params['learning_rate'],logger=logger)
            mean_private_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_private_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        b = datetime.now()
        temp = b-a
        print(temp)
        '''
        用于迭代 Progressive 模型
        '''
        for j in range(N_Participants):
            progressive_net_list[j] = copy.deepcopy(net_list[j])

        if epoch_index ==CommunicationEpoch-1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                private_dataset_dir = Dataset_Dir + private_dataset_name
                print(netname+'_'+private_dataset_name+'_'+private_dataset_dir)
                _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                                  train_bs=TrainBatchSize,
                                                  test_bs=TestBatchSize, dataidxs=None)
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
            acc_list.append(acc_epoch_list)

        if epoch_index % 5 == 0 or epoch_index == CommunicationEpoch - 1:
            mkdirs(Idea_Winter_Dir+ '/Performance_Analysis/' + Scenario)
            mkdirs(Idea_Winter_Dir+ '/Model_Storage/' + Scenario)
            mkdirs(Idea_Winter_Dir+ '/Performance_Analysis/' + Scenario + '/' + Ablation_Name)
            mkdirs(Idea_Winter_Dir+ '/Model_Storage/' + Scenario + '/' + Ablation_Name)
            mkdirs(Idea_Winter_Dir+ '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name)
            mkdirs(Idea_Winter_Dir+ '/Model_Storage/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name)

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save(Idea_Winter_Dir+ '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                    + '/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save(Idea_Winter_Dir+ '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                    + '/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save(Idea_Winter_Dir+ '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                    + '/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(),
                           Idea_Winter_Dir+ '/Model_Storage/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                           + '/' + netname + '_' + str(participant_index) + '_' + private_dataset_name + '.ckpt')
