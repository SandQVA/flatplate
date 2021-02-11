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
        self.threshold_angle = 10                             # threshold angle for B update in degrees
        self.c = config["CHORD"]                              # flat plate chord
        self.L = config["LENGTH"]                             # flat plate length
        self.t = config["THICKNESS"]                          # flat plate thickness
        self.S = self.c * self.L                              # flat plate surface
        self.rho_plate = 0.5 * 2                              # flat plate density (paper of 500g/m^2 density)
        self.m = self.rho_plate * self.L * self.c * self.t    # flate plate mass
        self.rho_air = 1.18415
        self.mr = (0.5 * self.rho_air * self.S) / self.m      # flat plate reduced mass
        self.g = -9.806                                       # gravity
        
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
        self.pitch_angle = action + np.random.normal(scale=self.config["ACTION_SIGMA"])
        
        # solve differential equation
        timearray=np.linspace(0,self.config["DELTA_TIME"],2)
        old_cartesian_state = self.get_state_in_absolute_cartesian_coordinates(old_polar_state)
        odestates = odeint(self.flatplate_equations,old_cartesian_state,timearray) # y = odeint(model, y0, t)
        new_cartesian_state = odestates[-1] # choose second state returned by odeint
        if new_cartesian_state[2] > 0:
            print('u is positive, the application has not been designed to be physically accurate in such cases')
        self.state = self.get_state_in_relative_polar_coordinates(new_cartesian_state)
        
        #print('self.state normalized', np.array2string(self.state, formatter={'float_kind':lambda x: "%.5f" % x}))
        denormalized_old_polar_state = self.denormalize_polar_state(old_polar_state)
        denormalized_state = self.denormalize_polar_state(self.state)

        # compute reward and check if the episode is over (done)
        reward = self.compute_reward(denormalized_old_polar_state, action, denormalized_state)
        won, lost = self.is_won_or_lost(denormalized_state)
        done = self.isdone(won, lost)
        if done:
            reward = self.update_reward_if_done(reward, won, lost)

        # save data for printing
        self.var_episode = self.var_episode + [list(new_cartesian_state) + list(self.pitch_angle/np.pi*180) + [reward]]
        
        return [self.state.copy(), reward, done, None]


    def reset(self, state=None):
        self.nb_ep +=1
        self.var_episode = []
        if np.mod(self.nb_ep,self.config["POINTB_CHANGE"]) == 0:
            self.update_B()

        if state==None:
            self.state = self.get_state_in_relative_polar_coordinates(self.cartesian_init)
        else:
            self.state = state

        return self.state.copy()


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
    def flatplate_equations(self, cartesian_state, t):
        u = cartesian_state[2]
        v = cartesian_state[3]
        V_norm = np.sqrt(u**2+v**2)

        flight_path_angle = - np.arctan2(-v, -u)
        alpha = self.pitch_angle - flight_path_angle
        if alpha > np.pi/2:
            print('the angle of attack is bigger than pi/2, the application has not been designed to be physically accurate in such cases')

        # Wang 2004 fitting careful, Wang data are calibrated for angles in degrees
        cl = 1.2 * np.sin(2*alpha*180/np.pi)
        cd = 1.4 - np.cos(2*alpha*180/np.pi)

        drag = self.mr * V_norm**2 * cd 
        lift = self.mr * V_norm**2 * cl
        
        dxdt = u
        dydt = v
       
        # Andrea's equations 
        #dudt = self.Drag/self.m 
        #dvdt = self.g + self.mr * V_norm**2 * alpha

        dudt = drag * np.cos(-flight_path_angle) - lift * np.sin(-flight_path_angle)
        dvdt = self.g + drag * np.sin(-flight_path_angle) + lift * np.cos(-flight_path_angle)
        
        computedstate = np.array([dxdt, dydt, dudt, dvdt]).astype(float)
        return computedstate


    def compute_reward(self, old_polar_state, action, new_polar_state):
        delta_rho = new_polar_state[0] - old_polar_state[0]
        #delta_abs_theta = np.abs(new_polar_state[1]) - np.abs(old_polar_state[1])
        #reward = -10000*delta_rho # go to goal
        #reward = (-100*delta_rho/self.rhoAB - 2*np.abs(new_polar_state[1])/np.pi)*10
        reward_rho = -100*delta_rho/self.rhoAB
        reward_theta = -20*np.abs(new_polar_state[1])/np.pi
        reward = reward_rho + reward_theta
        #reward = reward_rho
        #print('reward = reward rho + reward theta --> ', reward, ' = ', reward_rho, ' + ', reward_theta)

        return reward


    def update_reward_if_done(self, reward, won, lost):
        if won: reward += 100
        elif lost: reward += -100

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


    def print_won_or_lost(self, polar_state):
        won = False
        lost = False

        if np.abs(polar_state[0]/self.rhoAB) <= 10**(-3):
            won = True
            print('won')
        elif np.abs(polar_state[1]+self.phiA) >= np.pi/2.:
            lost = True
            print('lost')

        return won, lost


    # update B coordinates as required in the CFD config file
    def update_B(self):
        self.nb_pointB_change +=1

        self.xB = np.random.uniform(self.xA, self.config["XB"])
        self.yB = np.random.uniform(self.yA-self.diffyAB_init, self.yA+self.diffyAB_init)

        # keep iterating until the absolute angle between A and B is below the threshold angle
        while abs(np.arctan2(abs(self.yB-self.yA),abs(self.xB-self.xA))) > self.threshold_angle*np.pi/180:
            self.xB = np.random.uniform(self.xA, self.config["XB"])
            self.yB = np.random.uniform(self.yA-self.diffyAB_init, self.yA+self.diffyAB_init)

        print('Final absolute angle',abs(np.arctan2(abs(self.yB-self.yA),abs(self.xB-self.xA)))/np.pi*180, 'degrees')
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
        normalized_polar_state = self.normalize_polar_state(polar_state)

        return normalized_polar_state


    def get_state_in_absolute_cartesian_coordinates(self, polar_state):
        denormalized_polar_state = self.denormalize_polar_state(polar_state)
        rho = denormalized_polar_state[0]
        theta = denormalized_polar_state[1]
        rhoDot = denormalized_polar_state[2]
        thetaDot = denormalized_polar_state[3]
        phiP = theta + self.phiA

        x = self.xB + rho * np.cos(phiP)
        y = self.yB + rho * np.sin(phiP)

        u = rhoDot * np.cos(phiP) - thetaDot * np.sin(phiP)
        v = rhoDot * np.sin(phiP) + thetaDot * np.cos(phiP)
        cartesian_state = np.array([x, y, u, v])

        return cartesian_state


    def normalize_polar_state(self, state):
        normalized_state = np.zeros(4)
        normalized_state[0] = state[0]/self.rhoAB
        normalized_state[1] = state[1]/(np.pi/2)
        normalized_state[2] = state[2]/(self.rhoAB*(self.uA/self.BA[0]))
        normalized_state[3] = state[3]/((15*np.pi/180)/0.1)

        return normalized_state


    def denormalize_polar_state(self, state):
        denormalized_state = np.zeros(4)
        denormalized_state[0] = state[0]*self.rhoAB
        denormalized_state[1] = state[1]*(np.pi/2)
        denormalized_state[2] = state[2]*(self.rhoAB*(self.uA/self.BA[0]))
        denormalized_state[3] = state[3]*((15*np.pi/180)/0.1)

        return denormalized_state


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


    def plot_training_output(self, rewards, folder):
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
        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        plt.suptitle(folder.rsplit('/', 1)[1])
        plt.subplot(1,2,1)
        plt.title('Trajectories')
        plt.plot([self.xA,self.xB], [self.yA,self.yB], color='black', ls='--', label='Ideal path')
        plt.plot(xfirst, yfirst, label='first path')
        plt.plot(xlast, ylast, label='last path ep='+str(self.config['MAX_EPISODES']))
        plt.plot(xbest, ybest, label='best path ep='+str(best))
        plt.plot(xworst, yworst, label='worst path ep='+str(worst))
        plt.grid()
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)
        plt.legend(fontsize = 10, loc='best')

        plt.subplot(1,2,2)
        plt.title('Training reward')
        plt.plot(rewards, color='k')
        plt.grid()
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.savefig(f'{folder}/train_output.png')

    def plot_testing_output(self, rewards, folder):
        # TODO optimize this to avoid recomputing everything
        score = sum(rewards)/len(rewards) if rewards else 0
        xlast = np.trim_zeros(self.var_array[0,0,:], 'b')
        ylast = self.var_array[1,0,:len(xlast)]

        filex_train = f'{folder}/../x.csv'
        filey_train = f'{folder}/../y.csv'
        filereward_train = f'{folder}/../rewards.csv'
        xmatrix_train = np.loadtxt(filex_train, delimiter=";")
        ymatrix_train = np.loadtxt(filey_train, delimiter=";")
        rewardmatrix_train = np.loadtxt(filereward_train, delimiter=";")
        xlasttrain = np.trim_zeros(xmatrix_train[-1,:], 'b')
        ylasttrain = ymatrix_train[-1,:len(xlasttrain)]

        cumulative_reward = rewardmatrix_train.sum(axis=1)

        plt.cla()
        plt.figure(figsize=(10, 5))
        plt.suptitle(folder.rsplit('/', 2)[1])
        plt.subplot(1,2,1)
        plt.plot([self.xA,self.xB], [self.yA,self.yB], color='black', ls='--', label='Ideal path')
        plt.plot(xlasttrain, ylasttrain, label='last training path')
        plt.plot(xlast, ylast, label='test path')
        plt.grid()
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)
        plt.legend(fontsize = 10, loc='best')

        t='Rewards \n'+\
          f"   Test         = {score:9.0f}\n"+\
          f"   Train (last) = {cumulative_reward[-1]:9.0f}"

        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height

        ax = plt.subplot(1,2,2)
        ax.axis('off')
        ax.text(0.5*(left+right), 0.5*(bottom+top), t,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                color='k',
                fontsize=12)

        plt.savefig(f'{folder}/test_output.png')
