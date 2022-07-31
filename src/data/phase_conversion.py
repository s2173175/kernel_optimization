from matplotlib import container
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

def generate_dummy_data():
    phases = np.array([0.9, 0.4, 0.4, 0.9])

    phases_hist = np.array([phases])

    for i in range(1,1000):
        new_phase = phases + i*0.02
        new_phase = new_phase % 1
        phases_hist = np.vstack((phases_hist,new_phase))

    np.set_printoptions(suppress=True)


    return phases_hist

def generate_dummy_commands(phases,n=3):
    cmd = np.ones((len(phases),n ))*-3

    cmd_phase = np.hstack((cmd,phases))

    return cmd_phase


def plot_global(phases, swing_ratio=0.4, testing=False):
    plt.figure(figsize=(4,2))
    plt.plot(phases[:100,0], color='r')
    swing = np.where(phases[:100,0] < swing_ratio)[0]
    swing = np.split(swing, np.where(np.diff(swing) != 1)[0]+1)
    for r in swing:
        plt.axvspan(r[0], r[-1], facecolor='b', alpha=0.5)
    plt.show()

def global_to_norm(phases, swing_ratio=0.4, testing=False):

    norm = np.where(phases < swing_ratio, (phases/swing_ratio)+1, (phases-swing_ratio)/(1-swing_ratio))

    if testing:
        plt.figure(figsize=(4,2))
        plt.plot(norm[:100,0], color='r')
        swing = np.where(phases[:100,0] < swing_ratio)[0]
        swing = np.split(swing, np.where(np.diff(swing) != 1)[0]+1)
        for r in swing:
            plt.axvspan(r[0], r[-1], facecolor='b', alpha=0.5)
        plt.show()

    return norm


def global_to_sin(phases, swing_ratio=0.4, testing=False):

    sin_global = np.sin(phases*2*np.pi)

    if testing:
        plt.plot(sin_global[:200,0], color='r')

        swing = np.where(phases[:200,0] < swing_ratio)[0]
        swing = np.split(swing, np.where(np.diff(swing) != 1)[0]+1)
        for r in swing:
            plt.axvspan(r[0], r[-1], facecolor='b', alpha=0.5)
     
        plt.show()

    return


def global_to_norm_sin(phases, swing_ratio=0.4, testing=False):

    norm = global_to_norm(phases, swing_ratio=swing_ratio)
    sin_norm = np.sin(norm*np.pi)

    if testing:
        plt.figure(figsize=(4,2))
        plt.plot(sin_norm[:100,0], color='r')

        swing = np.where(phases[:100,0] < swing_ratio)[0]
        swing = np.split(swing, np.where(np.diff(swing) != 1)[0]+1)
        for r in swing:
            plt.axvspan(r[0], r[-1], facecolor='b', alpha=0.5)

        plt.show()
     

    return


def stack_sin_phase(cmd_phases, n=3, testing=False):

    phases = cmd_phases[:,-4:]

    init = phases[0]
    prepend = np.repeat(np.array([init]), n-1 , axis=0)

    phases = np.vstack((prepend, phases))

    stack_aligned = None
    for i in range(len(phases)-n+1):
        phase_hist = phases[i:i+n]
        phase_aligned = phase_hist.flatten('F')
        if stack_aligned is None:
            stack_aligned = np.array(phase_aligned)
        else:
            stack_aligned = np.vstack((stack_aligned,phase_aligned))

    cmd_phase_stacked_alligned = np.hstack((cmd_phases[:,:-4], stack_aligned))

    return cmd_phase_stacked_alligned


if __name__ == '__main__':

    phases = generate_dummy_data()
    # cmd_phase = generate_dummy_commands(phases=phases)

    # stack_sin_phase(cmd_phase)

    plot_global(phases, testing=True)
    global_to_norm(phases, testing=True)

    # global_to_sin(phases, testing=True)

    global_to_norm_sin(phases, testing=True)