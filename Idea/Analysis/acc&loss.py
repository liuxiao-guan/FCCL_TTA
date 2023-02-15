import sys
sys.path.append("..")
from Idea.utils_idea import mkdirs
from Idea.params import args_parser
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

args = args_parser()
CommunicationEpoch = args.CommunicationEpoch
N_Participants = args.N_Participants
Scenario = args.Scenario
Public_Dataset_Name = args.Public_Dataset_Name
Idea_Dir = args.Project_Dir+'Idea/'
Private_Dataset_Name_List = args.Private_Dataset_Name_List

Evaluation_Idea_Dict = {
    'Winter_Soldier':['Barlow_HSIC','Barlow_HSIC_ablation'],
    'Counterpart': ['Feddf', 'FedMD', 'FML', 'RCFL', 'FedMATCH'],
}

def visualize_all_acc(acc_all_list,method_list,epochs_num,n_partipants,save_path):
    method_acc_list = []
    for index, acc_list in enumerate(acc_all_list):
        batch_num = int(acc_list.shape[0]/epochs_num)
        epoch_acc_list = []
        for epoch_index in range(epochs_num+1):
            epoch_acc = (acc_list[epoch_index:epoch_index+batch_num][:])
            epoch_acc = epoch_acc.mean(axis=0).tolist()
            epoch_acc_list.append(round(mean(epoch_acc),2))
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

def visualize_loss(batch_loss,epochs_num,n_partipants,save_path):
    batch_num = int(batch_loss.shape[0]/epochs_num)
    epoch_loss_list = []
    for epoch_index in range(epochs_num):
        epoch_loss = (batch_loss[epoch_index*batch_num:(epoch_index+1)*batch_num][:])
        epoch_loss = epoch_loss.mean(axis=0).tolist()
        epoch_loss_list.append(round(mean(epoch_loss),4))
    x = np.arange(epochs_num)


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

            acc = np.load(acc_path)
            print('Evaluation Method: '+Method_Name+' Ablation Name: '+Ablation_Name)
            '''
            评估所有方法
            '''
            acc_list.append(acc)
            method_list.append(Ablation_Name)

            local_loss = np.load(local_loss_path,allow_pickle=True)
            visualize_loss(batch_loss=local_loss,epochs_num=CommunicationEpoch,n_partipants=N_Participants,save_path=save_local_loss_path)


    # visualize_all_acc(acc_all_list=acc_list,method_list=method_list,epochs_num=CommunicationEpoch,
    # n_partipants=N_Participants,save_path='/home/huangwenke/Wenke_Project/Marco_Polo/Idea/Winter_Soldier/Performance_Analysis/' + Scenario + '/' + 'Figure/All.png')
    #
