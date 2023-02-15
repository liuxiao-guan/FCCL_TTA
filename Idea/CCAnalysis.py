import sys
from Idea.utils_idea import mkdirs
sys.path.append("..")
from Network.utils_network import init_nets
from Dataset.utils_dataset import init_logs, get_dataloader,generate_public_data_idxs
from Idea.params import args_parser
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from numpy import *
import os


Evaluation_Idea_Dict = {
    'Winter_Soldier':['Barlow_HSIC']
}

args = args_parser()

Scenario = args.Scenario
Idea_Dir = args.Project_Dir+'Idea/'
Original_Path = args.Original_Path
Seed = args.Seed

N_Participants = args.N_Participants
TrainBatchSize = args.TrainBatchSize
TestBatchSize = args.TestBatchSize

Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Private_Net_Name_List = args.Private_Net_Name_List

'''
Digits scenario for large domain gap
'''
Public_Dataset_Name = args.Public_Dataset_Name
Public_Dataset_Dir = Dataset_Dir+Public_Dataset_Name
Public_Dataset_Length = args.Public_Dataset_Length
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)

def draw_heatmap(data,ablation_name,name):
    fig, ax = plt.subplots(figsize=(9, 9))
    # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
    # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
    sns.heatmap(
        pd.DataFrame(np.round(data, 2)),
        annot=False, vmax=1, vmin=0, xticklabels=False, yticklabels=False, cbar=False,square=False, cmap="Blues")
    # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
    #            square=True, cmap="YlGnBu")
    # ax.set_title('二维数组热力图', fontsize = 18)
    # ax.set_ylabel('Feature', fontsize=18)
    # ax.set_xlabel('Feature', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
    print(name)
    try:
        mkdirs('Analysis/CCAnalysis/'+Scenario)
    except OSError as error:
        print(error)
    try:
        mkdirs('Analysis/CCAnalysis/'+Scenario+'/'+ablation_name)
    except OSError as error:
        print(error)
    plt.savefig('Analysis/CCAnalysis/'+Scenario+'/'+ablation_name+'/'+name+'.pdf',bbox_inches='tight')

if __name__ =='__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = args.device_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info("Initalize Barlow Participants' Model")
    Barlow_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)

    logger.info("Initalize Origin Participants' Model")
    Origin_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_idxs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length)

    public_train_dl, _, _, _ = get_dataloader(dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir,
                                                            train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                            dataidxs=public_data_indexs)

    for key, value in Evaluation_Idea_Dict.items():
        for item in value:
            Method_Name = key
            Ablation_Name = item
            Method_Path = Idea_Dir + Method_Name + '/'
            logger.info('CC A: '+Method_Name+'-'+Ablation_Name)
            test_accuracy_list = []
            '''
            Load Model
            '''
            for particiapnt_index in range(N_Participants):
                '''
                加载 Barlow 模型
                '''
                network = Barlow_net_list[particiapnt_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                netname = Private_Net_Name_List[particiapnt_index]
                dataset_name = Private_Dataset_Name_List[particiapnt_index]
                network.load_state_dict(torch.load(Method_Path+'Model_Storage/' +Scenario+'/'+Ablation_Name+'/'+
                Public_Dataset_Name+'/'+ netname + '_' + str(particiapnt_index) + '_' +dataset_name+'.ckpt'))

    for index in range(N_Participants):
        network = Origin_net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[index]
        dataset_name = Private_Dataset_Name_List[index]
        '''
        加载 Original 模型
        '''
        network.load_state_dict(torch.load(Original_Path+'Model_Storage/'+ netname + '_' + str(index) + '_' +dataset_name+'.ckpt'))

    for batch_idx, (images, _) in enumerate(public_train_dl):
        images = images.to(device)
        linear_output_barlow_list  = []
        linear_output_origin_list  = []

        for participant_index in range(N_Participants):
            barlow_network = Barlow_net_list[participant_index]
            barlow_network = nn.DataParallel(barlow_network, device_ids=device_ids).to(device)
            barlow_network.eval()
            barlow_linear_output = barlow_network(x=images)
            linear_output_barlow_list.append(barlow_linear_output)

        for participant_index in range(N_Participants):
            origin_network = Origin_net_list[participant_index]
            origin_network = nn.DataParallel(origin_network, device_ids=device_ids).to(device)
            origin_network.eval()
            origin_linear_output = origin_network(x=images)
            linear_output_origin_list.append(origin_linear_output)

        '''
        画和平均结果的
        '''
        for participant_index in range(N_Participants):
            linear_output_barlow_avg = torch.mean(torch.stack(linear_output_barlow_list), 0)
            barlow_linear_output = linear_output_barlow_list[participant_index]
            z_1_bn = (barlow_linear_output - barlow_linear_output.mean(0)) / barlow_linear_output.std(0)
            z_2_bn = (linear_output_barlow_avg - linear_output_barlow_avg.mean(0)) / linear_output_barlow_avg.std(0)
            # empirical cross-correlation matrix
            c_barlow = z_1_bn.T @ z_2_bn
            # sum the cross-correlation matrix between all gpus
            c_barlow.div_(len(images))
            c_barlow_array = c_barlow.detach().cpu().numpy()
            draw_heatmap(c_barlow_array,Ablation_Name,'barrlow'+Private_Dataset_Name_List[participant_index])

            linear_output_origin_avg = torch.mean(torch.stack(linear_output_origin_list), 0)
            origin_linear_output = linear_output_origin_list[participant_index]
            z_1_bn = (origin_linear_output - origin_linear_output.mean(0)) / origin_linear_output.std(0)
            z_2_bn = (linear_output_origin_avg - linear_output_origin_avg.mean(0)) / linear_output_origin_avg.std(0)
            # empirical cross-correlation matrix
            c_origin = z_1_bn.T @ z_2_bn
            # sum the cross-correlation matrix between all gpus
            c_origin.div_(len(images))
            c_origin_array = c_origin.detach().cpu().numpy()
            draw_heatmap(c_origin_array,'origin','origin'+Private_Dataset_Name_List[participant_index])
            print('Something for Nothing')

        '''
        画两两结果的
        '''
        # for participant_index_i in range(N_Participants):
        #     for participant_index_j in range(N_Participants):
        #         if  participant_index_j > participant_index_i:
        #             barlow_linear_output_i = linear_output_barlow_list[participant_index_i]
        #             barlow_linear_output_j = linear_output_barlow_list[participant_index_j]
        #             z_1_bn = (barlow_linear_output_i - barlow_linear_output_i.mean(0)) / barlow_linear_output_i.std(0)
        #             z_2_bn = (barlow_linear_output_j - barlow_linear_output_j.mean(0)) / barlow_linear_output_j.std(0)
        #             # empirical cross-correlation matrix
        #             c_barlow = z_1_bn.T @ z_2_bn
        #             # sum the cross-correlation matrix between all gpus
        #             c_barlow.div_(len(images))
        #             c_barlow_array = c_barlow.detach().cpu().numpy()
        #             draw_heatmap(c_barlow_array,Ablation_Name,'barrlow'+Private_Dataset_Name_List[participant_index_i]+Private_Dataset_Name_List[participant_index_j])
        #
        #             origin_linear_output_i = linear_output_origin_list[participant_index_i]
        #             origin_linear_output_j = linear_output_origin_list[participant_index_j]
        #
        #             z_1_bn = (origin_linear_output_i - origin_linear_output_i.mean(0)) / origin_linear_output_i.std(0)
        #             z_2_bn = (origin_linear_output_j - origin_linear_output_j.mean(0)) / origin_linear_output_j.std(0)
        #             # empirical cross-correlation matrix
        #             c_origin = z_1_bn.T @ z_2_bn
        #             # sum the cross-correlation matrix between all gpus
        #             c_origin.div_(len(images))
        #             c_origin_array = c_origin.detach().cpu().numpy()
        #             draw_heatmap(c_origin_array,'origin','origin'+Private_Dataset_Name_List[participant_index_i]+Private_Dataset_Name_List[participant_index_j])

        break

