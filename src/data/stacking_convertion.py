from matplotlib import container
import numpy as np
import pandas as pd
import random

def generate_fake_data():

    example = np.arange(0,8,1)
    
    data = np.array([example])

    for i in range(1,1000):
        new_example = example + 0.1*i
        data = np.vstack((data,new_example))

    return data


def stack_inputs(all_inputs, n=3):

    init = all_inputs[0]
    prepend = np.repeat(np.array([init]), n-1 , axis=0)

    all_inputs = np.vstack((prepend, all_inputs))

    stack_aligned = None
    for i in range(len(all_inputs)-n+1):
        hist = all_inputs[i:i+n]
        aligned = hist.flatten('F')
        if stack_aligned is None:
            stack_aligned = np.array(aligned)
        else:
            stack_aligned = np.vstack((stack_aligned,aligned))


    return stack_aligned



def stack_output(all_outputs, n=3):

    stack_aligned = None
    for i in range(len(all_outputs)-n):
        future = all_outputs[i:i+n]
        aligned = future.flatten('F')
        if stack_aligned is None:
            stack_aligned = np.array(aligned)
        else:
            stack_aligned = np.vstack((stack_aligned,aligned))


    return stack_aligned



def stack_inputs_outputs(inputs, outputs, n, k):

    stacked_inputs = stack_inputs(inputs, n)
    stacked_output = stack_inputs(outputs, k)

    return stacked_inputs, stacked_output




 
if __name__ == '__main__':

    data = generate_fake_data()
    stack_inputs_outputs(data,data,3,4)