import pandas as pd
import matplotlib.pyplot as plt

import numpy as np



def plot_coms():

    x2 = pd.read_csv('./walking_cmd_com_x.csv', header=None)
    x3 = pd.read_csv('./walking_cmd_com3_x.csv', header=None)

    x2.columns = ['ang_cmd', 'for_cmd', 'side_cmd', 'com_1', 'com_2', 'com_3', 'l_vel_1', 'l_vel_2', 'l_vel_2', 'a_vel_1', 'a_vel_2', 'a_vel_3', 'p1', 'p2', 'p3', 'p4']
    x3.columns = ['ang_cmd', 'for_cmd', 'side_cmd', 'com_1', 'com_2', 'com_3', 'l_vel_1', 'l_vel_2', 'l_vel_2', 'a_vel_1', 'a_vel_2', 'a_vel_3', 'p1', 'p2', 'p3', 'p4']

    x2 = x2[['com_1', 'com_2', 'com_3']].values[10000:11000,:]
    x3 = x3[['com_1', 'com_2', 'com_3']].values[10000:11000,:]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x3[:,0], x3[:,1], x3[:,2], marker='^', alpha=0.5)
    ax.scatter(x2[:,0], x2[:,1], x2[:,2], marker='o', alpha=0.5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    return



def plot_coms_step_height():
    pd.set_option('precision', 4)
    data = pd.read_csv('./com_experiment.csv', header=None)

    data.columns = ['com1','com2','com3', 'step_h']


    fig = plt.figure()
    ax_05 = fig.add_subplot(4,3,1,projection='3d')
    ax_06 = fig.add_subplot(4,3,2,projection='3d')
    ax_07 = fig.add_subplot(4,3,3,projection='3d')

    ax_08 = fig.add_subplot(4,3,4,projection='3d')
    ax_09 = fig.add_subplot(4,3,5,projection='3d')
    ax_10 = fig.add_subplot(4,3,6,projection='3d')
    
    ax_11 = fig.add_subplot(4,3,7,projection='3d')
    ax_12 = fig.add_subplot(4,3,8,projection='3d')
    ax_13 = fig.add_subplot(4,3,9,projection='3d')

    ax_14 = fig.add_subplot(4,3,10,projection='3d')

    figs = [ax_05,ax_06,ax_07,ax_08,ax_09,ax_10,ax_11,ax_12,ax_13,ax_14]

    ids = np.arange(0.05,0.15, 0.01)

    print(ids)

    for i,ax in enumerate(figs):

        x = data.loc[np.isclose(data['step_h'], ids[i])]
        x = x[['com1','com2','com3']].values
        ax.scatter(x[:,0], x[:,1], x[:,2], marker='o', s=0.1)

   
        ax.axes.set_xlim3d(left=-0.06, right=0.035)
        ax.axes.set_ylim3d(bottom=-0.12, top=0.085)
        ax.axes.set_zlim3d(bottom=-1, top=-0.992)
   
        ax.set_title(str(ids[i])[:4])

 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    # plt.savefig('./com_variations.eps', format='eps', bbox_inches='tight')
    plt.show()

    return

def plot_coms_ride_height():
    pd.set_option('precision', 4)
    data = pd.read_csv('./com_experiment_ride.csv', header=None)

    data.columns = ['com1','com2','com3', 'step_h']


    fig = plt.figure()
    ax_05 = fig.add_subplot(4,3,1,projection='3d')
    ax_06 = fig.add_subplot(4,3,2,projection='3d')
    ax_07 = fig.add_subplot(4,3,3,projection='3d')

    ax_08 = fig.add_subplot(4,3,4,projection='3d')
    ax_09 = fig.add_subplot(4,3,5,projection='3d')
    ax_10 = fig.add_subplot(4,3,6,projection='3d')
    
    ax_11 = fig.add_subplot(4,3,7,projection='3d')
    ax_12 = fig.add_subplot(4,3,8,projection='3d')
    ax_13 = fig.add_subplot(4,3,9,projection='3d')

    ax_14 = fig.add_subplot(4,3,10,projection='3d')
    ax_15 = fig.add_subplot(4,3,11,projection='3d')

    figs = [ax_05,ax_06,ax_07,ax_08,ax_09,ax_10,ax_11,ax_12,ax_13,ax_14, ax_15]

    ids = np.arange(0.18, 0.30, 0.01)

    print(ids)

    for i,ax in enumerate(figs):

        x = data.loc[np.isclose(data['step_h'], ids[i])]
        x = x[['com1','com2','com3']].values
        ax.scatter(x[:,0], x[:,1], x[:,2], marker='o', s=0.1)

   
        ax.axes.set_xlim3d(left=-0.06, right=0.035)
        ax.axes.set_ylim3d(bottom=-0.12, top=0.085)
        ax.axes.set_zlim3d(bottom=-1, top=-0.992)
   
        ax.set_title(str(ids[i])[:4])

 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    # plt.savefig('./com_variations.eps', format='eps', bbox_inches='tight')
    plt.show()

    return


def com_variance():

    return


plot_coms_ride_height()
plot_coms_step_height()