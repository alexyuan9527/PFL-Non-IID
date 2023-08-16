import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


# with h5py.File("E:\\Documents\\GitHub\\PFL-Non-IID\\results\\mnist_FedAvg_test_0.h5", 'r') as f:
#     acc = f["rs_test_acc"]
#     acc_std = f["rs_test_acc_std"]
#     loss = f["rs_train_loss"]
#     print(acc)
    # for group in f.keys():
    #     print (group)
def plot_acc_loss(loss, acc, path):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()   # 共享x轴
 
    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("test-loss")
    par1.set_ylabel("test-accuracy")
 
    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.savefig(path)
    plt.show()



# csv_data = pd.read_csv("E:\\Documents\\GitHub\\PFL-Non-IID\\results\\mnist_FedAvg_test7456345_0.csv")
# plot_acc_loss(csv_data["train_loss"], csv_data["test_acc"], "1.png")