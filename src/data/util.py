from matplotlib import container
import numpy as np
import pandas as pd
import random


def walking_subset_csv(num_eps):

    x = pd.read_csv('./walking_cmd_v2_x.csv', header=None)
    y = pd.read_csv('./walking_cmd_v2_y.csv', header=None)

    
    x = x.loc[x[0] < num_eps]
    y = y.loc[y[0] < num_eps]

    x = x.drop(columns=[0])
    y = y.drop(columns=[0])

    x.to_csv(f'./sets/walking_cmd_{num_eps}_x.csv')
    y.to_csv(f'./sets/walking_cmd_{num_eps}_y.csv')


def walking_dest_csv():

    #    x["anglar_cmd"], \
    #         x["forward_speed_cmd"], \
    #         x["side_speed_cmd"], \
    #         x["phases"] \


    x = pd.read_csv('./walking_cmd_x.csv', header=None)
    y = pd.read_csv('./walking_cmd_y.csv', header=None)

  

    # # print(x.head(5))

    # jointA = list(range(0, 12))
    # jointV = list(range(12, 24))
    # gravity = list(range(24,27))
    # yaw_error = list(range(27,28))
    # b_vel = list(range(28,31))
    # target_distance = list(range(31,32))
    # feet_pose = list(range(32,44))
    # contact_state = list(range(44,48))
    # phase = list(range(48,52))

    # foot_targets = list(range(0,12))
    # foot_results = list(range(12,24))

    # drop_x = \
    #     jointV \
    #     + feet_pose \
    #     + jointA \
    #     + gravity \
    #     + b_vel \
    #     + contact_state \
    #     # + target_distance \
    #     # + yaw_error \
    #     # + phase

    # drop_y = foot_results

    # x.drop(x.columns[drop_x],axis=1,inplace=True)
    # y.drop(y.columns[drop_y],axis=1,inplace=True)



    # # print(x.head(5))

    x.to_csv('./sets/walking_cmd_x.csv')
    y.to_csv('./sets/walking_cmd_y.csv')


def walking_com_alts():

    # 'walking_cmd_2', ---
    # 'walking_cmd_3', ---
    # 'walking_cmd_2_com', ---
    # 'walking_cmd_3_com', ---
    # 'walking_cmd_2_L_A', ---

    x = pd.read_csv('./walking_cmd_com25_x.csv', header=None)
    y = pd.read_csv('./walking_cmd_com25_y.csv', header=None)

    x.columns = ['ang_cmd', 'for_cmd', 'side_cmd', 'com_1', 'com_2', 'com_3', 'l_vel_1', 'l_vel_2', 'l_vel_2', 'a_vel_1', 'a_vel_2', 'a_vel_3', 'p1', 'p2', 'p3', 'p4']

    drop = [ 'l_vel_1', 'l_vel_2', 'l_vel_2', 'a_vel_1', 'a_vel_2', 'a_vel_3']
    x = x.drop(columns=drop)

    # print(x.head())
    # x = x.drop(columns=[0])
    # y = y.drop(columns=[0])
    x.to_csv('./sets/walking_cmd_25_com_x.csv')
    y.to_csv('./sets/walking_cmd_25_com_y.csv')

    return

def stacked_sin_phase():
    """
    1) 
    """
    x = pd.read_csv('./walking_cmd_v2_x.csv', header=None)
    y = pd.read_csv('./walking_cmd_v2_y.csv', header=None)

    print(x.shape)
    print(y.shape)
    input()

    phases = x.iloc[:, -4:]
    sin_phase = -np.sin(np.array(phases)*np.pi)

    x.iloc[:, -4:] = sin_phase

    stacked_sin_phase = pd.DataFrame()

    for j in range(500):
        episode_x = x.loc[x[0] == j]
        print("episode: ", j, "-------------------------", episode_x.shape)
        sin_phases = episode_x.iloc[:, -4:].values

        lookback = 1
        stacked = np.array([])

        for i in range(len(sin_phases)):
  
            z = sin_phases[(i-lookback) if (i-lookback) > 0 else 0 : i+1 ]
            if len(z) < lookback + 1:
                z = np.repeat(z, 2, axis=0)
                
            p1 = z[:,0]
            p2 = z[:,1]
            p3 = z[:,2]
            p4 = z[:,3]
            
            final = np.hstack((p1,p2,p3,p4))
            if len(stacked) == 0:
                stacked = final
            else:
                stacked = np.vstack((stacked,final))



        df_ = episode_x.iloc[:, :-4] 
  
        df = pd.DataFrame(stacked, columns=list('abcdefgh'))
      
        result = pd.concat([df_, df.set_index(df_.index)], axis=1)


        stacked_sin_phase = pd.concat([stacked_sin_phase, result], ignore_index=True, sort=False)
       

    stacked_sin_phase = stacked_sin_phase.drop(columns=[0])
    y = y.drop(columns=[0])
    stacked_sin_phase.to_csv('./sets/walking_cmd_v2_x.csv')
    y.to_csv('./sets/walking_cmd_v2_y.csv')


    return



walking_com_alts()
# walking_subset_csv(100)
# walking_dest_csv()
# stacked_sin_phase()