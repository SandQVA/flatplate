# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time

folder = '/stck/s.berger/ml/cfd_sureli/results/DDPG/flateplate_2020-02-12_13-26-37'

with open(folder+'/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if not os.path.exists(folder+'/plots/'):
    os.mkdir(folder+'/plots/')
        
nb_episodes = config["MAX_EPISODES"]
#nb_episodes = 5

#----------------------------------- Environment settigs ------------------------------

# final position of the problem
xB = config["XB"]
yB = config["YB"]

# initial conditions of the problem
xA = config["XA"]
yA = config["YA"]
uA = config["UA"]
vA = config["VA"]


## --------------------------------------------------------
filetime = folder+'/time'+'.csv'
filex = folder+'/x'+'.csv'
filey = folder+'/y'+'.csv'
fileu = folder+'/u'+'.csv'
filev = folder+'/v'+'.csv'
fileactions = folder+'/actions'+'.csv'
filerewards = folder+'/rewards'+'.csv'

timematrix = np.loadtxt(filetime, delimiter=";")
xmatrix = np.loadtxt(filex, delimiter=";")
ymatrix = np.loadtxt(filey, delimiter=";")
umatrix = np.loadtxt(fileu, delimiter=";")
vmatrix = np.loadtxt(filev, delimiter=";")
actionsmatrix = np.loadtxt(fileactions, delimiter=";")
rewardsmatrix = np.loadtxt(filerewards, delimiter=";")

cumulative_reward = rewardsmatrix.sum(axis=1)


#xlasttrain = np.trim_zeros([-1,:], 'b')
#ylasttrain = ymatrix_train[-1,:len(xlasttrain)]


reds = plt.get_cmap("cool")

for i in range(nb_episodes):
    fig = plt.figure(num=1,figsize=(18,6))
    plt.cla()

    plt.subplot(1, 3, 1)
    j=i
    #for j in range(i):
        #plt.plot(xmatrix[:,j],ymatrix[:,j],label='', color=reds(j*(1/config["MAX_EPISODES"]) ), linewidth=0.5)
    plt.plot(np.trim_zeros(xmatrix[j,:], 'b'),ymatrix[j,:len(np.trim_zeros(xmatrix[j,:], 'b'))], label='', linewidth=1)
    plt.plot([xA,xB],[yA,yB],label='Ideal path', color='black', ls='--')
    plt.text(0.055, 0.027, 'r='+str(cumulative_reward[j])[:6], ha='center',fontsize = 20)
    plt.grid()
    plt.title('Training trajectories', fontsize=18)
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('y (m)', fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim([0.025, 0.065])
    plt.ylim([0.025, 0.045])
    plt.legend(fontsize = 14)

    plt.subplot(1, 3, 2)
    plt.plot(cumulative_reward[:i],label='', linewidth=1)
    plt.grid()
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Rewards', fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim([0., config["MAX_EPISODES"]])
    plt.ylim([1, 450])
    #plt.legend(fontsize = 14)
 
    plt.subplot(1, 3, 3)
    j=i
    #for j in range(i):
    plt.plot(timematrix[:len(np.trim_zeros(actionsmatrix[j,:]))+1], np.trim_zeros(actionsmatrix[j,:], 'b'),label='', linewidth=1)
    plt.grid()
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Angle (°)', fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim([timematrix[0], timematrix[-1]])
    plt.ylim([-16, 16])
    #plt.legend(fontsize = 14)
 
    plt.savefig(folder+'/plots/paths_'+str(i)+'.png')
    plt.close()
    fig.canvas.flush_events()
    time.sleep(1)
    #plt.show()
