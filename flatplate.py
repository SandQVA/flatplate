# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:17:59 2019

@author: andrea
"""



import numpy as np
from scipy.integrate import odeint


class FlatPlateModel:
    
    def __init__(self,initialconditions,finalposition,config):
        
        self.config = config
        
        self.xA = initialconditions[0]
        self.yA = initialconditions[1]
        self.A = np.array[self.xA,self.yA]
        self.uA = initialconditions[2]
        self.vA = initialconditions[3]
        self.xB = finalposition[0]
        self.yB = finalposition[1]
        self.B = np.array[self.xB,self.yB]
        self.rhoAB = np.linalg.norm(self.A-self.B)
        
        self.currentstate = np.array([self.xA,self.yA,self.uA,self.vA])
        
        self.Drag = 0                                   #not considering drag forces
        self.c = 0.1                                    #flat plate chord
        self.L = 1                                      #flat plate length
        self.t = 0.01                                   #flat plate thickness
        self.S = self.c*self.L
        self.rho_plate = 0.5*2                          #flat plate density (paper of 500g/m^2 density)
        self.m = self.rho_plate*self.L*self.c*self.t    #flate plate mass
        self.rho_air = 1.18415
        self.mr = (self.rho_air*self.S*np.pi)/self.m
        self.g = -9.806                                 #gravity taking into account the reference system
        

    def get_state_in_relative_polar_coordinates(self,state):
    	# TODO rhoDot, thetaDot
    	BP = state[0:2]-self.B
        BA = self.A-self.B
    	rho = np.linalg.norm(BP)
    	theta = np.acos (np.dot(BP,BA) / (rho * self.rhoAB))
    	return rho,theta,rhoDot,thetaDot

    # differential equations system for flat plate
    def flatplate(self,variables,t):
        #no time dependance on time for ODE
        x = variables[0]
        y = variables[1]
        u = variables[2]
        v = variables[3]
        
        V = np.sqrt(u**2+v**2)
        alphainduced = np.arctan2(v,u)
        if alphainduced <= np.pi and alphainduced >= 0:
            alphainduced = -(np.pi - np.abs(alphainduced))
        else:
            alphainduced = np.pi - np.abs(alphainduced)
        
        #set of differential equations
        dxdt = u
        dydt = v
        
        dudt = self.Drag/self.m 
        dvdt = self.g + self.mr * V**2 * (self.alpha + alphainduced) 
        
        computedstate = [dxdt, dydt, dudt, dvdt]
        return computedstate
    
    #compute next state, reward and done or not
    def step(self,action):
        #alpha is the action
        self.alpha = action
        old_polar_state = get_state_in_relative_polar_coordinates(self.currentstate)
        #if some noise is wanted to be added to 'model' turbulence effects
        self.alpha = action + np.random.normal(scale=config["ACTION_SIGMA"])
        
        #time vector with initial and final point of the step
        timearray=np.linspace(0,self.config["DELTA_TIME"],2)
        #y = odeint(model, y0, t)
        odestates = odeint(self.flatplate,self.currentstate,timearray)
        #as odeint will return two different states, choose the second one
        self.currentstate = odestates[-1]
        new_polar_state = get_state_in_relative_polar_coordinates(self.currentstate)
        
        #compute reward and done
        reward = self.compute_reward(old_polar_state, action, new_polar_state)
        done = self.isdone()
        
        # must return new state, reward, done (boolean)
        # print(newstate)
        return [new_polar_state, reward, done]

    #compute reward
    def compute_reward(self, old_polar_state, action, new_polar_state):
    	# TODO vérifier les signes dans le reward model

        #------------- Still working on tuning a good reward function ------------
        # the aim of the reward function is to have a more positive value if the followed
        #path is closer to the straight line at each step of the episode
        
#        k1 = 100
#        k2 = 5*180/np.pi
##        reward = 1-k1*(self.rho/self.rhoAB * (1+k2*abs(self.theta)))
#        reward = 1-k1*(self.rho/self.rhoAB + k1*k2*abs(self.theta))
##        reward = 1-k1*(self.rho/self.rhoAB+k1*k2*self.theta+2*(1-self.rho/self.rhoAB)*k2*self.theta)
        k1 = 100
        k2 = 5*180/np.pi
        k3 = 180/np.pi
        
        #reward = 1-k1*(rho/self.rhoAB + k1*k2*abs(theta) + k3*(1-rho/self.rhoAB)*abs(theta))
        delta_rho = new_polar_state[0] - old_polar_state[0]
        delta_abs_theta = np.abs(new_polar_state[1]) - np.abs(old_polar_state[1])
        reward = -delta_rho # go to goal
        #reward = -delta_rho - delta_abs_theta # go to goal along the AB line

        #reward normalisation according to the maximum number of steps
        #reward = reward/self.config["MAX_STEPS"]
        return reward
    
    
    #check if done
    def isdone(self):
        done = False
        #done if the final point is almost reached (error due to discretisation is considered)
        #or if point B x coordinate is almost reached
        polar_state = get_state_in_relative_polar_coordinates(self.currentstate)
        if np.abs(polar_state[0]/self.rhoAB) <= 10**(-3) or np.abs(polar_state[1]) >= np.pi/2.
            done = True
        return done
    
    #reset the model with initial conditions
    def reset(self, state=None):
    	if state=None:
	        self.currentstate = np.array([self.xA, self.yA, self.uA, self.vA])
	    else:
	    	self.currentstate = state
        return get_state_in_relative_polar_coordinates(self.currentstate)
        