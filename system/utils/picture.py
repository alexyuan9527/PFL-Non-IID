import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.pyplot import MultipleLocator


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


def plot_acc_loss2(loss, acc, std, path):
    l = int(len(loss)/2)
    fig = plt.figure(figsize=(21,5))
    # plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    
    ax1 = fig.add_subplot(131)
    ax1.plot(range(l), loss[::2], label="loss (glo)", c="tomato")
    ax1.plot(range(l), loss[1::2], label="loss (per)", c="red")
    ax1.set_xlabel("Global rounds")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(range(l))
    ax1.legend(loc=1)  # 1->rightup corner, 2->leftup corner, 3->leftdown corner, 4->rightdown corner, 5->rightmid
    
    ax2 = fig.add_subplot(132)
    ax2.plot(range(l), acc[::2], label="accuracy (glo)", c="lime")
    ax2.plot(range(l), acc[1::2], label="accuracy (per)", c="green")
    ax2.set_xlabel("Global rounds")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(range(l))
    ax2.legend(loc=4)
    
    ax3 = fig.add_subplot(133)
    ax3.plot(range(l), std[::2], label="std(acc) (glo)", c="aqua")
    ax3.plot(range(l), std[1::2], label="std(acc) (per)", c="deepskyblue")
    ax3.set_xlabel("Global rounds")
    ax3.set_ylabel("std(Accuracy)")
    ax3.set_xticks(range(l))
    ax3.legend(loc=5)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.draw()
    plt.savefig(path)
    plt.show()

# csv_data = pd.read_csv("E:\\Documents\\GitHub\\PFL-Non-IID\\results\\mnist_FedAvg_test7456345_0.csv")
# plot_acc_loss(csv_data["train_loss"], csv_data["test_acc"], "1.png")