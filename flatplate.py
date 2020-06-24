# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:17:59 2019 by Andrea
Last modified jun 7 2020 by Sandrine
"""

import numpy as np
from scipy.integrate import odeint
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FlatPlate:
    
    def __init__(self,config):
        
        self.config = config
       
        # initial and target conditions from config file
        self.xA = config["XA"]
        self.yA = config["YA"]
        self.A = np.array([self.xA,self.yA])
        self.uA = config["UA"]
        self.vA = config["VA"]
        self.xB = config["XB"]
        self.yB = config["YB"] 
        self.B = np.array([self.xB,self.yB])
        self.rhoAB = np.linalg.norm(self.A-self.B)
        self.diffyAB_init = abs(self.yA-self.yB)

        self.BA = self.A-self.B
        self.phiA = np.arctan2(self.BA[1],self.BA[0])

        # some parameters
        self.threshold_angle = 10                       # threshold angle for B update
        self.Drag = 0                                   # drag forces (set to zero)
        self.c = 0.1                                    # flat plate chord
        self.L = 1                                      # flat plate length
        self.t = 0.02                                   # flat plate thickness
        self.S = self.c*self.L                          # flat plate surface
        self.rho_plate = 0.5*2                          # flat plate density (paper of 500g/m^2 density)
        self.m = self.rho_plate*self.L*self.c*self.t    # flate plate mass
        self.rho_air = 1.18415
        self.mr = (self.rho_air*self.S*np.pi)/self.m    # flat plate reduced mass
        self.g = -9.806                                 # gravity
        
        # state initialisation
        self.cartesian_init = np.array([self.xA,self.yA, self.uA, self.vA])
        self.state = self.get_state_in_relative_polar_coordinates(self.cartesian_init)

        # attributs needed by the rl code or gym wrappers
        self.action_space = collections.namedtuple('action_space', ['low', 'high', 'shape'])(-15/180*np.pi, 15/180*np.pi, (1,))
        self.action_size = 1
        self.observation_space = collections.namedtuple('observation_space', ['shape'])(self.cartesian_init.shape)
        self.reward_range = None
        self.metadata = None

        # B coords array (size depending on update number)
        self.nb_ep = 0
        self.nb_pointB_change = 0
        self.B_array = np.zeros([self.config["MAX_EPISODES"]//self.config["POINTB_CHANGE"]+1, 2])
        self.B_array[self.nb_pointB_change, :] = self.B

        # dt_array and var_array initialisation (for plotting purposes only, not required by the application)    
        self.var_episode = [0]
        self.variables = ['x', 'y', 'u', 'v', 'actions', 'rewards']
        self.dt_array = np.array([i*config["DELTA_TIME"] for i in range(config["MAX_STEPS"]+1)])
        self.var_array = np.zeros([len(self.variables), config["MAX_EPISODES"],config["MAX_STEPS"]+1])
        # fill array with initial state
        for i in range(len(self.cartesian_init)):
            self.var_array[i,:,0] = self.cartesian_init[i]

 
    def step(self,action):
        old_polar_state = self.state
        self.alpha = action + np.random.normal(scale=self.config["ACTION_SIGMA"])
        
        # solve differential equation
        timearray=np.linspace(0,self.config["DELTA_TIME"],2)
        old_cartesian_state = self.get_state_in_absolute_cartesian_coordinates(old_polar_state)
        odestates = odeint(self.flatplate,old_cartesian_state,timearray) # y = odeint(model, y0, t)
        new_cartesian_state = odestates[-1] # choose second state returned by odeint
        self.state = self.get_state_in_relative_polar_coordinates(new_cartesian_state)
       
        # compute reward and check if the episode is over (done)
        reward = self.compute_reward(old_polar_state, action, self.state)
        won, lost = self.is_won_or_lost(self.state)
        done = self.isdone(won, lost)
        if done:
            reward = self.update_reward_if_done(reward, won, lost)

        # save data for printing
        self.var_episode = self.var_episode + [list(new_cartesian_state) + list(self.alpha/np.pi*180) + [reward]]
        
        return [self.state, reward, done, None]


    def reset(self, state=None):
        if state==None:
            self.state = self.get_state_in_relative_polar_coordinates(self.cartesian_init)
        else:
            self.state = state

        self.nb_ep +=1
        self.var_episode = []
        if np.mod(self.nb_ep,self.config["POINTB_CHANGE"]) == 0:
            self.update_B()

        return self.state


    # defined in order to mimic a gym environment
    def render(self, mode='human'):
        pass


    # defined in order to mimic a gym environment
    def close(self):
        pass


    # defined in order to mimic a gym environment
    def seed(self):
        pass


    # differential equations system for flat plate
    def flatplate(self, cartesian_state, t):
        u = cartesian_state[2]
        v = cartesian_state[3]
        
        V = np.sqrt(u**2+v**2)
        alphainduced = np.arctan2(v,u)
        # TO DO check
        if alphainduced <= np.pi and alphainduced >= 0:
            alphainduced = -(np.pi - np.abs(alphainduced))
        else:
            alphainduced = np.pi - np.abs(alphainduced)
        
        dxdt = u
        dydt = v
        
        dudt = self.Drag/self.m 
        dvdt = self.g + self.mr * V**2 * (self.alpha + alphainduced) 
        
        computedstate = np.array([dxdt, dydt, dudt, dvdt]).astype(float)
        return computedstate


    def compute_reward(self, old_polar_state, action, new_polar_state):
        delta_rho = new_polar_state[0] - old_polar_state[0]
        delta_abs_theta = np.abs(new_polar_state[1]) - np.abs(old_polar_state[1])
        #reward = -10000*delta_rho # go to goal
        reward = (-100*delta_rho/self.rhoAB - 2*np.abs(new_polar_state[1])/np.pi)*10


        return reward


    def update_reward_if_done(self, reward, won, lost):
        if won: reward += 1000
        elif lost: reward += -1000

        return reward

    
    def isdone(self, won, lost):
        done = False
        if won or lost:
            done = True

        return done
    

    def is_won_or_lost(self, polar_state):
        won = False
        lost = False

        if np.abs(polar_state[0]/self.rhoAB) <= 10**(-3):
            won = True
        elif np.abs(polar_state[1]+self.phiA) >= np.pi/2.:
            lost = True

        return won, lost


    def update_B(self):
        self.nb_pointB_change +=1

        self.xB = np.random.uniform(self.xA, self.config["XB"])
        self.yB = np.random.uniform(self.yA-self.diffyAB_init, self.yA+self.diffyAB_init)

        # keep iterating until the absolute angle between A and B is below 10 degrees
        while abs(np.arctan2(abs(self.yB-self.yA),abs(self.xB-self.xA))) > self.threshold_angle*np.pi/180:
            self.xB = np.random.uniform(self.xA, self.config["XB"])
            self.yB = np.random.uniform(self.yA-self.diffyAB_init, self.yA+self.diffyAB_init)

        print('Final absolute angle',abs(np.arctan2(abs(self.yB-self.yA),abs(self.xB-self.xA)))/np.pi*180)
        print('Final point coordinates: (',self.xB,self.yB,')')

        self.B = np.array([self.xB,self.yB])
        self.rhoAB = np.linalg.norm(self.A-self.B)
        self.BA = self.A-self.B
        self.phiA = np.arctan2(self.BA[1],self.BA[0])
        
        self.B_array[self.nb_pointB_change, :] = self.B 


# Below are only utility functions ------------------------------------------

    def get_state_in_relative_polar_coordinates(self, cartesian_state):
        BP = cartesian_state[0:2]-self.B
        rho = np.linalg.norm(BP)
        phiP = np.arctan2(BP[1],BP[0])
        theta = phiP - self.phiA

        u = cartesian_state[2]
        v = cartesian_state[3]

        rhoDot = u * np.cos(phiP) + v * np.sin(phiP)
        thetaDot = - u * np.sin(phiP) + v * np.cos(phiP)
        polar_state = np.array([rho, theta, rhoDot, thetaDot])

        return polar_state


    def get_state_in_absolute_cartesian_coordinates(self, polar_state):
        rho = polar_state[0]
        theta = polar_state[1]
        rhoDot = polar_state[2]
        thetaDot = polar_state[3]
        phiP = theta + self.phiA

        x = self.xB + rho * np.cos(phiP)
        y = self.yB + rho * np.sin(phiP)

        u = rhoDot * np.cos(phiP) - thetaDot * np.sin(phiP)
        v = rhoDot * np.sin(phiP) + thetaDot * np.cos(phiP)
        cartesian_state = np.array([x, y, u, v])

        return cartesian_state


    def fill_array_tobesaved(self):
        for i in range(len(self.variables)):
            for k in range(len(self.var_episode)):
                self.var_array[i, self.nb_ep-1, k+1] = self.var_episode[k][i]


    def print_array_in_files(self, folder):
        for i, var in enumerate(self.variables):
            filename = folder+'/'+var+'.csv'
            np.savetxt(filename, self.var_array[i,:,:], delimiter=";")
 
        filename = folder+'/Bcoordinates.csv'
        np.savetxt(filename, self.B_array, delimiter=";")

        filename = folder+'/time.csv'
        np.savetxt(filename, self.dt_array, delimiter=";")


    def plot_some_training_paths(self, folder):
        # TODO optimize this to avoid recomputing everything
        xfirst = np.trim_zeros(self.var_array[0,0,:], 'b')
        yfirst = self.var_array[1,0,:len(xfirst)]
        xlast = np.trim_zeros(self.var_array[0,-1,:], 'b')
        ylast = self.var_array[1,-1,:len(xlast)] 

        cumulative_reward = self.var_array[5,:,:].sum(axis=1)

        best = np.argmax(cumulative_reward)
        xbest = np.trim_zeros(self.var_array[0,best,:], 'b')
        ybest = self.var_array[1,best,:len(xbest)]

        worst = np.argmin(cumulative_reward)
        xworst = np.trim_zeros(self.var_array[0,worst,:], 'b')
        yworst = self.var_array[1,worst,:len(xworst)]

        plt.cla()
        plt.title(folder.rsplit('/', 1)[1])
        plt.plot([self.xA,self.xB], [self.yA,self.yB], color='black', ls='--', label='Ideal path')
        plt.plot(xfirst, yfirst, label='first path')
        plt.plot(xlast, ylast, label='last path ep='+str(self.config['MAX_EPISODES']))
        plt.plot(xbest, ybest, label='best path ep='+str(best))
        plt.plot(xworst, yworst, label='worst path ep='+str(worst))
        plt.grid()
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)
        plt.legend(fontsize = 14)
        plt.savefig(f'{folder}/some_trajectories.png')


    def plot_some_testing_paths(self, folder):
        # TODO optimize this to avoid recomputing everything
        xlast = np.trim_zeros(self.var_array[0,9,:], 'b')
        ylast = self.var_array[1,9,:len(xlast)]

        filex_train = f'{folder}/../x.csv'
        filey_train = f'{folder}/../y.csv'
        xmatrix_train = np.loadtxt(filex_train, delimiter=";")
        ymatrix_train = np.loadtxt(filey_train, delimiter=";")
        xlasttrain = np.trim_zeros(xmatrix_train[-1,:], 'b')
        ylasttrain = ymatrix_train[-1,:len(xlasttrain)]
 
        plt.cla()
        plt.title(folder.rsplit('/', 1)[1])
        plt.plot([self.xA,self.xB], [self.yA,self.yB], color='black', ls='--', label='Ideal path')
        plt.plot(xlasttrain, ylasttrain, label='last training path')
        plt.plot(xlast, ylast, label='test path')
        plt.grid()
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)
        plt.legend(fontsize = 14)
        plt.savefig(f'{folder}/test_trajectory.png')
