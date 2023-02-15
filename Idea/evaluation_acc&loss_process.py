import sys
sys.path.append("..")
sys.path.append(".")
from Idea.utils_idea import mkdirs
from Idea.params import args_parser
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from prettytable import PrettyTable

'''
Evaluate contrastive loss, local loss, and accuracy
'''

args = args_parser()
CommunicationEpoch = args.CommunicationEpoch
N_Participants = args.N_Participants

Evaluation_Idea_Dict = {
    'Winter_Soldier':['Barlow_Weight_l2_frozen','Barlow_HSIC',
                      'Barlow&Progressive','Barlow_HSIC_ablation',
                      'Barlow&Progressive_Ablation','Barlow_Weight_l2_frozen_Ablation'],
    'Counterpart': ['Feddf', 'FedMD', 'FML', 'RCFL', 'FedMATCH'],
}

# Evaluation_Idea_Dict = {
#     'Winter_Soldier':['Barlow_HSIC','Barlow_HSIC_ablation','Barlow_HSIC_WtIni'],
#     'Counterpart': ['Feddf', 'FedMD', 'FML', 'RCFL', 'FedMATCH'],
# }

Evaluation_Idea_Dict = {
    'Winter_Soldier':['Barlow_HSIC','Barlow_HSIC_ablation','Barlow_HSIC_OnlyIntra','Barlow_HSIC_OnlyInter','Barlow_HSIC_IM'],
    'Counterpart': ['Feddf', 'FedMD', 'FML', 'RCFL', 'FedMATCH'],
}

#
# Evaluation_Idea_Dict = {
#     'Winter_Soldier':['Barlow_HSIC_homo'],
#     'Counterpart': ['FedAvg','FedMD_homo','Feddf_homo','RCFL_homo','FedMATCH_homo','FML_homo'],
# }

Evaluation_Idea_Dict = {
    'Winter_Soldier':['Barlow_HSIC','Barlow_HSIC_FISL_NTKD','Barlow_HSIC_FISL','Barlow&Progressive'],
    #'Counterpart': ['FedAvg','FedMD_homo','Feddf_homo','RCFL_homo','FedMATCH_homo','FML_homo'],
}

Scenario = args.Scenario
Public_Dataset_Name = args.Public_Dataset_Name
Idea_Dir = args.Project_Dir+'Idea/'
Private_Dataset_Name_List = args.Private_Dataset_Name_List


def visualize_acc(acc_list,epochs_num,n_partipants,save_path):
    batch_num = int(acc_list.shape[0]/epochs_num)
    epoch_acc_list = []
    for epoch_index in range(epochs_num+1):
        epoch_acc = (acc_list[epoch_index:epoch_index+batch_num][:])
        epoch_acc = epoch_acc.mean(axis=0).tolist()
        epoch_acc_list.append(epoch_acc)
    x = np.arange(epochs_num+1)
    labels = []
    for index in range(0, n_partipants):
        plt.plot(x, [i[index] for i in epoch_acc_list])
        labels.append(Private_Dataset_Name_List[index])
    plt.ylim(50, 90)
    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
    plt.savefig(save_path,bbox_inches ='tight')
    plt.close()

def visualize_all_acc(acc_all_list,method_list,epochs_num,n_partipants,save_path):
    method_acc_list = []
    for index, acc_list in enumerate(acc_all_list):
        batch_num = int(acc_list.shape[0]/epochs_num)
        epoch_acc_list = []
        for epoch_index in range(epochs_num+1):
            epoch_acc = (acc_list[epoch_index:epoch_index+batch_num][:])
            epoch_acc = epoch_acc.mean(axis=0).tolist()
            epoch_acc_list.append(mean(epoch_acc))
        method_acc_list.append(epoch_acc_list)
    x = np.arange(epochs_num+1)
    labels = []
    for index in range(0, len(method_list)):
        plt.plot(x, method_acc_list[index])
        labels.append(method_list[index])
    plt.ylim(65, 90)
    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
    plt.savefig(save_path,bbox_inches ='tight')
    plt.close()

def calculate_acc(acc_list,epochs_num,n_partipants):
    batch_num = int(acc_list.shape[0]/epochs_num)
    epoch_acc_list = []
    for epoch_index in range(epochs_num+1):
        epoch_acc = (acc_list[epoch_index:epoch_index+batch_num][:])
        epoch_acc = epoch_acc.mean(axis=0).tolist()
        epoch_acc_list.append(epoch_acc)
    acc_ini_list = []
    acc_high_list = []
    acc_final_list = []
    for index in range(0, n_partipants):
        participant_acc_list = []
        for item in epoch_acc_list:
            participant_acc_list.append(item[index])
        acc_ini_list.append(round(participant_acc_list[0],3))
        acc_high_list.append(round(max(participant_acc_list),3))
        acc_final_list.append(round(mean(participant_acc_list[-3:]),3))
    return acc_ini_list,acc_high_list,acc_final_list

def visualize_loss(batch_loss,epochs_num,n_partipants,save_path):
    batch_num = int(batch_loss.shape[0]/epochs_num)
    epoch_loss_list = []
    for epoch_index in range(epochs_num):
        epoch_loss = (batch_loss[epoch_index*batch_num:(epoch_index+1)*batch_num][:])
        epoch_loss = epoch_loss.mean(axis=0).tolist()
        epoch_loss_list.append(epoch_loss)
    x = np.arange(epochs_num)
    labels = []
    for index in range(0, n_partipants):
        plt.plot(x, [i[index] for i in epoch_loss_list])
        labels.append('The Participant '+str(index)+' th')
    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)
    plt.savefig(save_path,bbox_inches ='tight')
    plt.close()

if __name__ == '__main__':
    acc_list = []
    method_list = []
    print(Scenario+Public_Dataset_Name)
    for key, value in Evaluation_Idea_Dict.items():
        for item in value:
            Method_Name = key
            Ablation_Name = item

            Method_Path = Idea_Dir + Method_Name + '/'
            collaborative_loss_path = Method_Path + 'Performance_Analysis/' + Scenario + '/' \
                                      + Ablation_Name + '/'+ Public_Dataset_Name +'/collaborative_loss.npy'
            local_loss_path = Method_Path + 'Performance_Analysis/' + Scenario + '/' \
                              + Ablation_Name + '/'+ Public_Dataset_Name + '/local_loss.npy'
            acc_path = Method_Path + 'Performance_Analysis/' + Scenario + '/' \
                       + Ablation_Name + '/'+ Public_Dataset_Name + '/acc.npy'

            save_path = Method_Path + 'Performance_Analysis/' + Scenario + '/' + 'Figure/' \
                        +  Ablation_Name + '/'+ Public_Dataset_Name
            save_acc_path = save_path + '/acc_figure.png'
            save_collaborative_loss_path = save_path + '/collaborative_loss_figure.png'
            save_local_loss_path = save_path + '/local_loss_figure.png'

            mkdirs(save_path)
            acc = np.load(acc_path)
            print('Evaluation Method: '+Method_Name+' Ablation Name: '+Ablation_Name)
            '''
            评估所有方法
            '''
            acc_list.append(acc)
            method_list.append(Ablation_Name)

            visualize_acc(acc_list=acc,epochs_num=CommunicationEpoch,n_partipants=N_Participants,save_path=save_acc_path)

            collaborative_loss = np.load(collaborative_loss_path)
            visualize_loss(batch_loss=collaborative_loss,epochs_num=CommunicationEpoch,n_partipants=N_Participants,save_path=save_collaborative_loss_path)

            local_loss = np.load(local_loss_path,allow_pickle=True)
            visualize_loss(batch_loss=local_loss,epochs_num=CommunicationEpoch,n_partipants=N_Participants,save_path=save_local_loss_path)

    visualize_all_acc(acc_all_list=acc_list,method_list=method_list,epochs_num=CommunicationEpoch,
    n_partipants=N_Participants,save_path='/home/guanxiaoliu/project/FCCL_PLUS/Idea/Winter_Soldier/Performance_Analysis/' + Scenario + '/' + 'Figure/All.png')

    pt_high = PrettyTable()
    pt_high.field_names = ['Method name','Ablation Name']+Private_Dataset_Name_List+['AVG']
    pt_final = PrettyTable()
    pt_final.field_names = ['Method name','Ablation Name']+Private_Dataset_Name_List+['AVG']

    for key, value in Evaluation_Idea_Dict.items():
        for item in value:
            Method_Name = key
            Ablation_Name = item

            Method_Path = Idea_Dir + Method_Name + '/'
            acc_path = Method_Path + 'Performance_Analysis/' + Scenario + '/' \
                       + Ablation_Name + '/' + Public_Dataset_Name + '/acc.npy'
            acc = np.load(acc_path)
            method_list  = [Method_Name,Ablation_Name]
            acc_ini_list,acc_high_list,acc_final_list = calculate_acc(acc_list=acc,epochs_num=CommunicationEpoch,n_partipants=N_Participants)
            avg_high = round(mean(acc_high_list),3)
            avg_final = round(mean(acc_final_list),3)
            acc_high_list.append(avg_high)
            acc_final_list.append(avg_final)
            pt_high.add_row(method_list+acc_high_list)
            pt_final.add_row(method_list+acc_final_list)
    avg_init = round(mean(acc_ini_list),3)
    acc_ini_list.append(avg_init)
    pt_high.add_row(['Without','-']+acc_ini_list)
    pt_final.add_row(['Without','-']+acc_ini_list)

    print('***********************************')
    print('The High Accuracy')
    print(pt_high.get_string(sortby=("AVG"), reversesort=True))
    print('***********************************')
    print('The Final Accuracy')
    print(pt_final.get_string(sortby=("AVG"), reversesort=True))
    print('***********************************')

    for private_data_index in range(N_Participants):
        private_data_name = Private_Dataset_Name_List[private_data_index]
        temp = ['Method name','Ablation Name']+[private_data_name]
        print(pt_final.get_string(fields=temp,sortby=(private_data_name), reversesort=True))
        print('***********************************')

