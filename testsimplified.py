# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:20:49 2019

@author: andrea
"""

import sys
sys.path.extend(['../commons/'])

import os
import argparse
import yaml
try:
    import roboschool
except ModuleNotFoundError:
    pass
import torch

import numpy as np

from model import Model
from utils import get_latest_dir
from flatplate import FlatPlateModel

parser = argparse.ArgumentParser(description='Test DDPG')
parser.add_argument('--no_render', action='store_false', dest="render",
                    help='Display the tests')
parser.add_argument('--gif', action='store_true', dest="gif",
                    help='Save a gif of a test')
parser.add_argument('-n', '--nb_tests', default=10, type=int, dest="nb_tests",
                    help="Number of evaluation to perform.")
parser.add_argument('-f', '--folder', default=None, type=str, dest="folder",
                    help="Folder where the models are saved")
args = parser.parse_args()

if args.folder is None:
    args.folder = os.path.join('runsFlatPlate/', get_latest_dir('runsFlatPlate/'))

with open(os.path.join(args.folder, 'configuration.yaml'), 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cpu')

if not os.path.exists(args.folder+'/test/'):
    os.mkdir(args.folder+'/test/')


## --------------------------------- Environment settigs ------------------------------

# final position of the problem
xB = config["XB"]
yB = config["YB"]

finalstate=np.array([xB, yB])

# initial conditions of the problem
xA = config["XA"]
yA = config["YA"]
uA = config["UA"]
vA = config["VA"]
initialconditions = np.array([xA,yA,uA,vA])

LOW_BOUND = -15*np.pi/180
HIGH_BOUND = 15*np.pi/180
STATE_SIZE = initialconditions.shape[0]
ACTION_SIZE = 1

## --------------------------------------------------------

# Create gym environment
env = FlatPlateModel(initialconditions,finalstate,config)

# Creating neural networks and loading models
model = Model(device, STATE_SIZE, ACTION_SIZE,args.folder, config)
model.load()
print("\033[91m\033[1mModel loaded from ", args.folder, "\033[0m")

n_ep = config["TEST_EPISODES"]

rewards = []
try:    
    
    xmatrix = np.zeros([config["MAX_STEPS"]+1,config["TEST_EPISODES"]+1])
    ymatrix = np.zeros([config["MAX_STEPS"]+1,config["TEST_EPISODES"]+1])
    umatrix = np.zeros([config["MAX_STEPS"]+1,config["TEST_EPISODES"]+1])
    vmatrix = np.zeros([config["MAX_STEPS"]+1,config["TEST_EPISODES"]+1])
    
    actionsmatrix = np.zeros([config["MAX_STEPS"]+1,config["TEST_EPISODES"]+1])
    rewardsmatrix = np.zeros([config["MAX_STEPS"]+1,config["TEST_EPISODES"]+1])
    
    for i in range(config["MAX_STEPS"]+1):
        xmatrix[i][0] = i*config["DELTA_TIME"]
        ymatrix[i][0] = i*config["DELTA_TIME"]
        umatrix[i][0] = i*config["DELTA_TIME"]
        vmatrix[i][0] = i*config["DELTA_TIME"]
        actionsmatrix[i][0] = i*config["DELTA_TIME"]
        rewardsmatrix[i][0] = i*config["DELTA_TIME"]
    
    columnepisode = 1
    
    for i in range(n_ep):
        
        rowstep = 0
        
        print('Episode number',i+1,'out of',n_ep,'keep waiting...')
        state = env.reset()
        reward = 0
        done = False
        step = 0
        while not done and step < config["MAX_STEPS"]:
            
            rowstep += 1
            
            action = model.select_action(state)
            action = np.clip(action, LOW_BOUND, HIGH_BOUND)
            state, r, done = env.step(action)
            reward += r
            step += 1

            cartesian_state = env.get_state_in_absolute_cartesian_coordinates(state) 
            xmatrix[rowstep][columnepisode] = cartesian_state[0]
            ymatrix[rowstep][columnepisode] = cartesian_state[1]
            umatrix[rowstep][columnepisode] = cartesian_state[2]
            vmatrix[rowstep][columnepisode] = cartesian_state[3]
            
            actionsmatrix[rowstep][columnepisode] = action[0]*180/np.pi
            rewardsmatrix[rowstep][columnepisode] = r
            
        rewards.append(reward)
        print('Episode reward:',reward)
        columnepisode += 1
        
        
except KeyboardInterrupt:
    pass
finally:
    
    np.savetxt(str(args.folder)+'/test/xplate.csv', xmatrix[:,0:n_ep], delimiter=";")
    np.savetxt(str(args.folder)+'/test/yplate.csv', ymatrix[:,0:n_ep], delimiter=";")
    np.savetxt(str(args.folder)+'/test/uplate.csv', umatrix[:,0:n_ep], delimiter=";")
    np.savetxt(str(args.folder)+'/test/vplate.csv', vmatrix[:,0:n_ep], delimiter=";")
    np.savetxt(str(args.folder)+'/test/actions.csv', actionsmatrix[:,0:n_ep], delimiter=";")
    np.savetxt(str(args.folder)+'/test/rewards.csv', rewardsmatrix[:,0:n_ep], delimiter=";")
    
    if rewards:
        score = sum(rewards)/len(rewards)
    else:
        score = 0

print(f"Average score : {score}")
#score = model.evaluate(env,n_ep=args.nb_tests)#, render=args.render, gif=args.gif)
