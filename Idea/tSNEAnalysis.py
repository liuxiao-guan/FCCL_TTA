import sys
sys.path.append("..")
from Network.utils_network import init_nets
from Dataset.utils_dataset import init_logs, get_dataloader
from Idea.params import args_parser
import matplotlib.pyplot as plt
from sklearn import manifold
import torch
import torch.nn as nn
import numpy as np

import os

Evaluation_Idea_Dict = {
    'Winter_Soldier':['Barlow_HSIC_ablation','Barlow_HSIC']
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

Public_Dataset_Name = args.Public_Dataset_Name
Public_Dataset_Dir = Dataset_Dir+Public_Dataset_Name
Public_Dataset_Length = args.Public_Dataset_Length
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)


def tsne_network(net,dataloader,ablation_name,model_name,classes):
    with torch.no_grad():
        logit_list = []
        label_list = []
        for images, labels in dataloader:
            images = images.to(device)
            outputs = net(images)
            outputs = outputs.cpu().numpy().tolist()
            labels = labels.numpy().tolist()
            logit_list.extend(outputs)
            label_list.extend(labels)
        logit_array = np.array(logit_list).flatten().reshape(-1, Output_Channel)
        label_array = np.array(label_list).flatten()
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(logit_array)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        scatter  = plt.scatter(X_norm[:,0], X_norm[:, 1], cmap=plt.cm.tab10,c=label_array)
        # plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.xticks([])
        plt.yticks([])
        try:
            mkdirs('Analysis/tSNEAnalysis/' + ablation_name)
        except OSError as error:
            print(error)
        plt.savefig('Analysis/tSNEAnalysis/' + ablation_name + '/' + 'tsne_'+model_name+'.pdf',bbox_inches='tight')
        plt.close()


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
    Barlow_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List)

    logger.info("Initalize Origin Participants' Model")
    Origin_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List)

    logger.info("Initalize Global Participants' Model")
    Global_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List)

    for key, value in Evaluation_Idea_Dict.items():
        for item in value:
            Method_Name = key
            Ablation_Name = item
            Method_Path = Idea_Dir + Method_Name + '/'
            logger.info('tSNE '+Method_Name+'-'+Ablation_Name)

            '''
            加载 Barlow 模型
            '''
            for particiapnt_index in range(N_Participants):
                network = Barlow_net_list[particiapnt_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                netname = Private_Net_Name_List[particiapnt_index]
                dataset_name = Private_Dataset_Name_List[particiapnt_index]
                network.load_state_dict(torch.load(Method_Path+'Model_Storage/' +Scenario+'/'+Ablation_Name+'/'+
                Public_Dataset_Name+'/'+ netname + '_' + str(particiapnt_index) + '_' +dataset_name+'.ckpt'))

            '''
            加载 Original 模型
            '''
            for index in range(N_Participants):
                network = Origin_net_list[index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                netname = Private_Net_Name_List[index]
                dataset_name = Private_Dataset_Name_List[index]
                network.load_state_dict(torch.load(
                    Original_Path + 'Model_Storage/' + netname + '_' + str(index) + '_' + dataset_name + '.ckpt'))

            '''
            加载 Global 模型
            '''
            for index in range(N_Participants):
                network = Global_net_list[index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                netname = Private_Net_Name_List[index]
                dataset_name = Private_Dataset_Name_List[index]
                network.load_state_dict(torch.load(
                    Original_Path + 'Model_Storage/' + netname + '_Global_' + dataset_name + '.ckpt'))

            '''
            测试 Barlow 模型
            '''
            for index in range(N_Participants):
                for target_index in range(N_Participants):
                    participant_dataset_name = Private_Dataset_Name_List[index]
                    target_dataset_name = Private_Dataset_Name_List[target_index]
                    target_dataset_dir = Dataset_Dir + target_dataset_name
                    _, test_dl, _, _ = get_dataloader(
                        dataset=target_dataset_name, datadir=target_dataset_dir, train_bs=TrainBatchSize,
                        test_bs=TestBatchSize,
                        dataidxs=None)
                    network = Barlow_net_list[index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    netname = Private_Net_Name_List[index]
                    tsne_network(net=network, dataloader=test_dl, ablation_name=Ablation_Name,model_name=participant_dataset_name
                                                                                                         +'_'+netname+'_'+target_dataset_name,
                                 classes=Private_Dataset_Classes)

                    network = Origin_net_list[index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    netname = Private_Net_Name_List[index]
                    tsne_network(net=network, dataloader=test_dl, ablation_name='Origin',model_name=participant_dataset_name
                                                                                                         +'_'+netname+'_'+target_dataset_name,
                                 classes=Private_Dataset_Classes)

                    network = Global_net_list[index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    netname = Private_Net_Name_List[index]
                    tsne_network(net=network, dataloader=test_dl, ablation_name='Global',model_name=participant_dataset_name
                                                                                                         +'_'+netname+'_'+target_dataset_name,
                                 classes=Private_Dataset_Classes)
