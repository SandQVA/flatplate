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

folder = f'runsFlatPlate/FlatPlate_2020-01-17_16-44-42'
        
with open(folder+'/configuration.yaml', 'w') as file:
    yaml.dump(config, file)
        
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
            
filex = folder+'/test/xplate'+'.csv'
filey = folder+'/test/yplate'+'.csv'
filex_train = folder+'/xplate'+'.csv'
filey_train = folder+'/yplate'+'.csv'
fileu = folder+'/test/uplate'+'.csv'
filev = folder+'/test/vplate'+'.csv'
fileactions = folder+'/test/actions'+'.csv'
filerewards = folder+'/test/rewards'+'.csv'
xmatrix = np.loadtxt(filex, delimiter=";")
ymatrix = np.loadtxt(filey, delimiter=";")
xmatrix_train = np.loadtxt(filex_train, delimiter=";")
ymatrix_train = np.loadtxt(filey_train, delimiter=";")
umatrix = np.loadtxt(fileu, delimiter=";")
vmatrix = np.loadtxt(filev, delimiter=";")
actionsmatrix = np.loadtxt(fileactions, delimiter=";")
rewardsmatrix = np.loadtxt(filerewards, delimiter=";")

print(xmatrix.shape)

reds = plt.get_cmap("cool")

plt.cla()
#for i in range(1,xmatrix.shape[1]):    
    #plt.plot(xmatrix[:,config["RANDOM_EPISODES"]],ymatrix[:,config["RANDOM_EPISODES"]],label='First AI path')
plt.plot([xA,xB],[yA,yB],label='Ideal path', color='black', ls='--')
plt.plot(xmatrix_train[:,-1],ymatrix_train[:,-1],'.', label='last train', linewidth=1, markersize=2)
plt.plot(xmatrix[:,-1],ymatrix[:,-1],'.', label='test', linewidth=1, markersize=2, color='red')
plt.grid()
plt.title('Trajectory modification', fontsize=18)
plt.xlabel('x position (m)', fontsize=14)
plt.ylabel('y position (m)', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([0.025, 0.065])
plt.ylim([0.025, 0.045])
plt.legend(fontsize = 14)
plt.savefig(folder+'/test/trajectory_test.png')

plt.cla()
for i in range(1,xmatrix.shape[1]):    
    plt.plot(actionsmatrix[:-5,0],actionsmatrix[:-5,i],label='', linewidth=1)
plt.grid()
plt.title('Performed actions', fontsize=18)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Angle (?)', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
#plt.legend(fontsize = 14)
plt.savefig(folder+'/test/actions_test.png')

