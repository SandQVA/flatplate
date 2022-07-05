# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

class FlatPlate:

    def __init__(self,config):
        
        self.config = config
       
        # initial and target conditions from config file
        self.xA = 0.
        self.yA = 0.
        self.uA = config["UA"]
        self.vA = config["VA"]

        self.B_array = []
        Btype=config['BTYPE']
        # B is fixed, values are given in the config file 
        if Btype=='fixed':
            self.xB = self.config["XB"]
            self.yB = self.config["YB"]
            self.B = np.array([self.xB,self.yB])
            self.B_array.append(self.B)
            A = np.array([self.xA,self.yA])
            self.rho0 = np.linalg.norm(A-self.B)
        # B is set randomly (in a given range) at each episode
        elif Btype=='random': 
            self.xB = 0.
            self.yB = 0. 
            self.B = np.array([self.xB,self.yB])
            self.rho0 = self.config["DISTANCE_RANGE"][1]
        else: print('Btype not correctly defined')

        # some parameters
        self.nb_ep = 0
        self.done = False
        self.pitch = 0.
        self.pitchrate = 0.
        self.max_pitchrate = config["MAX_PITCHRATE"]
        self.c = config["CHORD"]                                  # flat plate chord
        self.L = config["LENGTH"]                                 # flat plate length
        self.t = config["THICKNESS"]                              # flat plate thickness
        self.S = self.c * self.L                                  # flat plate surface
        self.rho_air = 1.0
        self.rho_plate = config["DENSITY_RATIO"] * self.rho_air   # flat plate density
        self.m = self.rho_plate * self.L * self.c * self.t        # flate plate mass
        self.mr = 0.5 * self.rho_air * self.S
        self.g = -9.806                                           # gravity

        # state initialisation
        self.cartesian_init = np.array([self.xA, self.yA, self.uA, self.vA])
        self.state = np.zeros(6)

        # attributs needed by the rl code or gym wrappers
        self.action_space = collections.namedtuple('action_space', ['low', 'high', 'shape'])(-self.max_pitchrate, self.max_pitchrate, (1,))
        self.action_size = 1
        self.observation_space = collections.namedtuple('observation_space', ['shape'])(self.state.shape)
        self.reward_range = None
        self.metadata = None

        # dt_array and var_array initialisation (for postproc and plotting purposes only, not required by the application)    
        self.cfd_var_episode = []
        self.cfd_var_names = ['x', 'y', 'u', 'v', 'pitch_cfd']
        self.cfd_var_array = np.zeros([len(self.cfd_var_names), config["MAX_EPISODES"], config["MAX_STEPS"]*config["CFD_ITERATIONS"]+1])
        self.rl_var_episode = []
        self.rl_var_names = ['rho', 'sintheta', 'costheta', 'rhodot', 'thetadot', 'pitch', 'action', 'reward']
        self.rl_var_array = np.zeros([len(self.rl_var_names), config["MAX_EPISODES"], config["MAX_STEPS"]])


    def step(self,action):
        old_polar_state = self.state
        old_cartesian_state = self.get_state_in_absolute_cartesian_coordinates(old_polar_state)
        self.pitchrate = action + np.random.normal(scale=self.max_pitchrate*self.config["ACTION_SIGMA"])

        new_cartesian_state = self.cfd_iterations(old_cartesian_state, self.config["CFD_ITERATIONS"])
       
        self.state = self.get_state_in_normalized_polar_coordinates(new_cartesian_state, self.pitch)
        #print('state', np.array2string(self.state, formatter={'float_kind':lambda x: "%.5f" % x}))
 
        # compute reward and check if the episode is over (done)
        reward = self.compute_reward(old_polar_state, self.state)
        won, lost = self.is_won_or_lost(self.state)
        if won or lost:
            self.done = True
        if self.done:
            reward = self.update_reward_if_done(reward, won, lost)
        done = self.done

        # save data for printing
        self.rl_var_episode = self.rl_var_episode + [list(self.state) + [action] + [reward]]
        
        return [self.state.copy(), reward, done, None]


    def cfd_iterations(self, old_cartesian_state, nb_ite):
        for i in range(nb_ite):
            self.pitch = self.pitch + self.pitchrate * self.config["DELTA_TIME"]

            # solve differential equation
            timearray=np.linspace(0,self.config["DELTA_TIME"],2)
            odestates = odeint(self.flatplate_equations,old_cartesian_state,timearray) # y = odeint(model, y0, t)
            new_cartesian_state = odestates[-1] # choose second state returned by odeint
            if new_cartesian_state[2] > 0:
                print('u is positive')
                self.done = True
                break
            old_cartesian_state = new_cartesian_state

            # save data for printing
            self.cfd_var_episode = self.cfd_var_episode + [list(new_cartesian_state) + [self.pitch]]

        return new_cartesian_state 


    def reset(self, Btype='random'):
        self.nb_ep +=1
        self.rl_var_episode = []
        self.done = False
        self.pitch = 0.

        # B is fixed, values are given in the config file 
        if Btype=='fixed':
            pass # values defined in the class init

        # B is set randomly (in a given range) at each episode
        elif Btype=='random':
            self.update_B_random()
            self.B = np.array([self.xB,self.yB])
            self.B_array.append(self.B)

        # a fixed set of B positions is used
        elif Btype=='batch':
            print('not yet coded')
            quit()

        # define initial state according to the position of B
        self.state = self.get_state_in_normalized_polar_coordinates(self.cartesian_init, self.pitch)
        # fill array with initial state
        self.cfd_var_episode = [list(self.cartesian_init) + [self.pitch]]
        
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

        flight_path_angle = np.arctan2(-v, -u)
        alpha = - self.pitch + flight_path_angle
        if alpha > np.pi/2:
            print('the angle of attack is bigger than pi/2, the application has not been designed to be physically accurate in such cases')

        cl = 1.3433 * np.sin(2*alpha)
        cd = 1.5055 - 1.3509 * np.cos(2*alpha)

        drag = self.mr * V_norm**2 * cd 
        lift = self.mr * V_norm**2 * cl

        dxdt = u
        dydt = v
        dudt = ( drag * np.cos(flight_path_angle) - lift * np.sin(flight_path_angle) ) / self.m
        dvdt = self.g + ( drag * np.sin(flight_path_angle) + lift * np.cos(flight_path_angle) ) / self.m
 
        computedstate = np.array([dxdt, dydt, dudt, dvdt]).astype(float)

        return computedstate


    def compute_reward(self, old_polar_state, new_polar_state):
        if self.config["REWARD_TYPE"] == 'dense':
            delta_rho = new_polar_state[0] - old_polar_state[0]
            reward = -100*delta_rho
        elif self.config["REWARD_TYPE"] == 'sparse':
            reward = 0.0
        else:
            print('!!! please define reward !!!')

        return reward


    def update_reward_if_done(self, reward, won, lost):
        if won: reward += 10.0
        elif lost: reward += -10.0

        return reward

    
    def is_won_or_lost(self, polar_state):
        won = False
        lost = False

        if np.abs(polar_state[0]) <= 10**(-2):
            won = True
        elif polar_state[2] <= 0.:
            lost = True

        return won, lost


    def print_won_or_lost(self, polar_state):
        won = False
        lost = False

        if np.abs(polar_state[0]) <= 10**(-2):
            won = True
            print('won')
        elif polar_state[2] <= 0.:
            lost = True
            print('lost')
        else: print('lost')

        return won, lost


    # update B coordinates as required in the CFD config file
    def update_B_random(self):
        env_angle_range = np.array(self.config["ANGLE_RANGE"])/180*np.pi
        env_distance_range = np.array(self.config["DISTANCE_RANGE"])
        thetaA = -np.random.uniform(env_angle_range[0], env_angle_range[1])
        rhoAB = np.random.uniform(env_distance_range[0], env_distance_range[1])

        self.xB = rhoAB * np.cos(np.pi+thetaA)
        self.yB = rhoAB * np.sin(np.pi+thetaA)


# Below are only utility functions ------------------------------------------

    def get_state_in_normalized_polar_coordinates(self, cartesian_state, pitch):
        BP = cartesian_state[0:2]-self.B
        rho = np.linalg.norm(BP)
        theta = np.arctan2(BP[1],BP[0])

        u = cartesian_state[2]
        v = cartesian_state[3]

        rhoDot = u * np.cos(theta) + v * np.sin(theta)
        thetaDot = - u * np.sin(theta) + v * np.cos(theta)
        polar_state = np.array([rho, np.sin(theta), np.cos(theta), rhoDot, thetaDot, pitch])
        normalized_polar_state = self.normalize_polar_state(polar_state)

        return normalized_polar_state


    def get_state_in_absolute_cartesian_coordinates(self, polar_state):
        denormalized_polar_state = self.denormalize_polar_state(polar_state)
        rho = denormalized_polar_state[0]
        sintheta = denormalized_polar_state[1]
        costheta = denormalized_polar_state[2]
        theta = np.arccos(costheta) * np.sign(sintheta)
        
        rhoDot = denormalized_polar_state[3]
        thetaDot = denormalized_polar_state[4]

        x = self.xB + rho * np.cos(theta)
        y = self.yB + rho * np.sin(theta)

        u = rhoDot * np.cos(theta) - thetaDot * np.sin(theta)
        v = rhoDot * np.sin(theta) + thetaDot * np.cos(theta)
        cartesian_state = np.array([x, y, u, v])

        return cartesian_state


    def normalize_polar_state(self, state):
        normalized_state = np.zeros(6)
        normalized_state[0] = state[0]/self.rho0
        normalized_state[1] = state[1]
        normalized_state[2] = state[2]
        normalized_state[3] = state[3]/np.sqrt(self.uA**2+self.vA**2)
        normalized_state[4] = state[4]/(self.max_pitchrate/100)
        normalized_state[5] = state[5]

        return normalized_state


    def denormalize_polar_state(self, state):
        denormalized_state = np.zeros(6)
        denormalized_state[0] = state[0]*self.rho0
        denormalized_state[1] = state[1]
        denormalized_state[2] = state[2]
        denormalized_state[3] = state[3]*np.sqrt(self.uA**2+self.vA**2)
        denormalized_state[4] = state[4]*(self.max_pitchrate/100)
        denormalized_state[5] = state[5]

        return denormalized_state


    def fill_array_tobesaved(self):
        for i in range(len(self.cfd_var_names)):
            for k in range(len(self.cfd_var_episode)):
                self.cfd_var_array[i, self.nb_ep-1, k] = self.cfd_var_episode[k][i]

        for i in range(len(self.rl_var_names)):
            for k in range(len(self.rl_var_episode)):
                self.rl_var_array[i, self.nb_ep-1, k] = self.rl_var_episode[k][i]


    def print_array_in_files(self, folder):
        for i, var in enumerate(self.cfd_var_names):
            filename = folder+'/'+var+'.csv'
            np.savetxt(filename, self.cfd_var_array[i,:,:], delimiter=";")

        for i, var in enumerate(self.rl_var_names):
            filename = folder+'/'+var+'.csv'
            np.savetxt(filename, self.rl_var_array[i,:,:], delimiter=";")

        filename = folder+'/Bcoordinates.csv'
        np.savetxt(filename, self.B_array, delimiter=";")


    def plot_training_output(self, returns, eval_returns, freq_eval, folder):
        xlast = np.trim_zeros(self.cfd_var_array[0,-1,:], 'b')
        ylast = self.cfd_var_array[1,-1,:len(xlast)] 

        cumulative_reward = self.rl_var_array[7,:,:].sum(axis=1)

        best = np.argmax(cumulative_reward)
        xbest = np.trim_zeros(self.cfd_var_array[0,best,:], 'b')
        ybest = self.cfd_var_array[1,best,:len(xbest)]
        if len(self.B_array) == 1:
            Bbest = self.B_array[0]
        else: 
            Bbest = self.B_array[best]

        ep_eval = [i*freq_eval for i in range(1, len(eval_returns)+1)]

        plt.cla()
        plt.figure(figsize=(14, 5))
        plt.tight_layout()
        plt.suptitle(folder.rsplit('/', 1)[1])
        plt.subplot(1,3,2)
        plt.title('Trajectories')
        plt.plot([self.xA, self.xB], [self.yA, self.yB], color='black', ls='--', label='Ideal path')
        plt.plot(xlast, ylast, color='black', label='last path ep='+str(self.config['MAX_EPISODES']))
        plt.plot([self.xA, Bbest[0]], [self.yA, Bbest[1]], color='green', ls='--', label='Ideal path')
        plt.plot(xbest, ybest, color='green', label='best path ep='+str(best))
        plt.grid()
        #plt.axis('equal')
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)
        plt.legend(fontsize = 10, loc='best')

        plt.subplot(1,3,1)
        plt.title('Training and evaluation returns')
        plt.plot(returns, color='black', label='training')
        plt.plot(ep_eval, eval_returns, color='red', label='evaluation')
        plt.grid()
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Return', fontsize=14)
        plt.legend(loc=0)

        plt.subplot(1,3,3)
        plt.plot(np.transpose(self.B_array)[0], np.transpose(self.B_array)[1], '.', color='blue')
        plt.scatter(self.xA, self.yA, 150, color='black', zorder=1.0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'{folder}/train_output.png')


    def plot_testing_output(self, returns, folder):
        score = sum(returns)/len(returns) if returns else 0
        xlast = np.trim_zeros(self.cfd_var_array[0,0,:], 'b')
        ylast = self.cfd_var_array[1,0,:len(xlast)]

        plt.cla()
        plt.figure(figsize=(10, 5))
        plt.suptitle(folder.rsplit('/', 2)[1])
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.90, hspace = 0.6)

        plt.subplot(1,3,1)
        plt.title('Testing return for each episode')
        plt.plot(returns, 'o-', color='black', label='training')
        plt.grid()
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Return', fontsize=14)

        plt.subplot(1,3,2)
        n_cmap = len(returns)
        cmap = pl.cm.tab10(np.linspace(0,1,n_cmap))

        if len(self.B_array) == 1:
            plt.plot([self.xA,self.B_array[0][0]], [self.yA,self.B_array[0][1]], color='black', ls='--', label='Ideal path')
        else:
            for i in range(len(returns)):
                plt.plot([self.xA,self.B_array[i][0]], [self.yA,self.B_array[i][1]], color=cmap[i], ls='--', label='Ideal path')
        
        for i in range(len(returns)):
            x = np.trim_zeros(self.cfd_var_array[0,i,:], 'b')
            y = self.cfd_var_array[1,i,:len(x)]
            plt.plot(x,y, color=cmap[i], label='test path')
        plt.grid()
        plt.axis('equal')
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)

        plt.subplot(1,3,3)
        plt.plot(np.transpose(self.B_array[:len(returns)])[0], np.transpose(self.B_array[:len(returns)])[1], '.', color='blue')
        plt.scatter(self.xA, self.yA, 150, color='black', zorder=1.0)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(f'{folder}/test_output.png')
