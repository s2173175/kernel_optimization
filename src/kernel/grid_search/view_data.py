import pandas as pd
import matplotlib.pyplot as plt



def view_top_ten():
    data = pd.read_csv("./gs_walk_dest.csv", header=None)

    mean_group = data.groupby([0, 1, 2, 3, 4]).mean()
    std_group = data.groupby([0, 1, 2, 3, 4]).std()

    mean_values = mean_group[7].values
    std_values = std_group[7].values

    
    

    print(data.head(5))

    # print(data[7].nsmallest(5))

    # print(data[7].idxmin())


view_top_ten()