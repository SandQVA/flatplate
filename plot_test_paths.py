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

folder = f'runsFlatPlate/008_2'
        
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
fileu = folder+'/test/uplate'+'.csv'
filev = folder+'/test/vplate'+'.csv'
fileactions = folder+'/test/actions'+'.csv'
filerewards = folder+'/test/rewards'+'.csv'
xmatrix = np.loadtxt(filex, delimiter=";")
ymatrix = np.loadtxt(filey, delimiter=";")
umatrix = np.loadtxt(fileu, delimiter=";")
vmatrix = np.loadtxt(filev, delimiter=";")
actionsmatrix = np.loadtxt(fileactions, delimiter=";")
rewardsmatrix = np.loadtxt(filerewards, delimiter=";")

print(xmatrix.shape)

reds = plt.get_cmap("cool")

plt.cla()
for i in range(1,xmatrix.shape[1]):    
    #plt.plot(xmatrix[:,config["RANDOM_EPISODES"]],ymatrix[:,config["RANDOM_EPISODES"]],label='First AI path')
    plt.plot(xmatrix[:-5,i],ymatrix[:-5,i],label='', linewidth=1)
plt.plot([xA,xB],[yA,yB],label='Ideal path', color='black', ls='--')
plt.grid()
plt.title('Trajectory modification', fontsize=18)
plt.xlabel('x position (m)', fontsize=14)
plt.ylabel('y position (m)', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
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

