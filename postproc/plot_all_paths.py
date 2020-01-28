# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:21:33 2019

@author: andrea
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from model import Model
from flatplate import FlatPlateModel


with open('configuration.yaml', 'r') as file:
    config = yaml.safe_load(file)


#folder = f'runsFlatPlate/{config["MODEL"].split("-")[0]}_{current_time}'
folder = f'runsFlatPlate/'
if not os.path.exists(folder+'/plots/'):
    os.mkdir(folder+'/plots/')
        
with open(folder+'/configuration.yaml', 'w') as file:
    yaml.dump(config, file)
        
nb_episodes = config["MAX_EPISODES"]
#nb_episodes = 51

#----------------------------------- Environment settigs ------------------------------

# final position of the problem
xB = config["XB"]
yB = config["YB"]

finalstate=[xB, yB]

# initial conditions of the problem
xA = config["XA"]
yA = config["YA"]
uA = config["UA"]
vA = config["VA"]
initialconditions = [xA,yA,uA,vA]

## --------------------------------------------------------
            
filex = folder+'/xplate'+'.csv'
filey = folder+'/yplate'+'.csv'
fileu = folder+'/uplate'+'.csv'
filev = folder+'/vplate'+'.csv'
fileactions = folder+'/actions'+'.csv'
filerewards = folder+'/rewards'+'.csv'
xmatrix = np.loadtxt(filex, delimiter=";")
ymatrix = np.loadtxt(filey, delimiter=";")
umatrix = np.loadtxt(fileu, delimiter=";")
vmatrix = np.loadtxt(filev, delimiter=";")
actionsmatrix = np.loadtxt(fileactions, delimiter=";")
rewardsmatrix = np.loadtxt(filerewards, delimiter=";")

print(xmatrix.shape)
print(config["MAX_EPISODES"]+1)

reds = plt.get_cmap("cool")


for i in range(1,nb_episodes):    
    #plot some paths (first random, first computed by AI, last path, ideal path)
    plt.cla()
    #plt.plot(xmatrix[:,config["RANDOM_EPISODES"]],ymatrix[:,config["RANDOM_EPISODES"]],label='First AI path')
    for j in range(1,i):
        #plt.plot(xmatrix[:,j],ymatrix[:,j],label='', color=reds(j*(1/config["MAX_EPISODES"]) ), linewidth=0.5)
        plt.plot(xmatrix[:-5,j],ymatrix[:-5,j],'.', label='', linewidth=1, markersize='1')
    plt.plot([xA,xB],[yA,yB],label='Ideal path', color='black', ls='--')
    plt.grid()
    plt.title('Trajectory modification', fontsize=18)
    plt.xlabel('x position (m)', fontsize=14)
    plt.ylabel('y position (m)', fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim([0.025, 0.065])
    plt.ylim([0.025, 0.045])
    plt.legend(fontsize = 14)
    plt.savefig(folder+'/plots/trajectory_'+str(i)+'.png')

for i in range(1,nb_episodes):    
    #plot some paths (first random, first computed by AI, last path, ideal path)
    plt.cla()
    #plt.plot(xmatrix[:,config["RANDOM_EPISODES"]],ymatrix[:,config["RANDOM_EPISODES"]],label='First AI path')
    for j in range(1,i):
        #plt.plot(xmatrix[:,j],ymatrix[:,j],label='', color=reds(j*(1/config["MAX_EPISODES"]) ), linewidth=0.5)
        plt.plot(actionsmatrix[:-5,0],actionsmatrix[:-5,j],label='', linewidth=1)
    plt.grid()
    plt.title('Performed actions', fontsize=18)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Angle (?)', fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.legend(fontsize = 14)
    plt.savefig(folder+'/plots/actions_'+str(i)+'.png')

