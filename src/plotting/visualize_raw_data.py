import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_standing_walk_gait():
    data = np.load("../data/standing_walk.npy", allow_pickle=True)
    
    episode = data[1]['y']

    print(episode[0]["foot_pose"])

    z = []
    for timestep in range(len(episode)):
        z.append(episode[timestep]["foot_pose"][0][2])

    plt.plot(z)
    plt.show()


visualize_standing_walk_gait()