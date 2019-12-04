# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:17:59 2019

@author: andrea
"""



import numpy as np
from scipy.integrate import odeint

from vpython import vector


class FlatPlateModel:
    
    def __init__(self,initialconditions,finalposition,config):
        
        self.config = config
        
        self.initialconditions = initialconditions
        self.xA = initialconditions[0]
        self.yA = initialconditions[1]
        self.uA = initialconditions[2]
        self.vA = initialconditions[3]
        
        self.currentstate = [self.xA,self.yA,self.uA,self.vA]
        
        self.xB = finalposition[0]
        self.yB = finalposition[1]
        
        
        self.rhoAB = np.sqrt((self.xB-self.xA)**2 + (self.yB-self.yA)**2)
        
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
            alphainduced = -(np.pi - abs(alphainduced))
        else:
            alphainduced = np.pi - abs(alphainduced)
        
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
        
        #if some noise is wanted to be added to 'model' turbulence effects
        #self.alpha = action + np.random.normal(scale=config["EXPLO_SIGMA"], size=ACTION_SIZE)/5
        
        #time vector with initial and final point of the step
        timearray=np.linspace(0,self.config["DELTA_TIME"],2)
        #y = odeint(model, y0, t)
        odestates = odeint(self.flatplate,self.currentstate,timearray)
        #as odeint will return two different states, choose the second one
        newstate=odestates[-1]
        self.currentstate = newstate
        
        #compute reward and done
        reward = self.compute_reward()
        done = self.isdone()
        
        # must return new state, reward, done (boolean)
        # print(newstate)
        return [newstate, reward, done]

    #compute reward
    def compute_reward(self):
        #compute the vector of the 
        vectorAB = vector(self.xB-self.xA,self.yB-self.yA,0)
        #compute the vector from the current point P to the final point B
        vectorPB = vector(self.xB-self.currentstate[0],self.yB-self.currentstate[1],0)
        
        #compute the angle between initial vector and the current straigth line 
        #to the final point B (deviation from the straight line to the final point B)
        self.theta = vectorAB.diff_angle(vectorPB)
        
        #distance from initial point A to final point B
        self.rhoAB = np.sqrt((self.xB-self.xA)**2 + (self.yB-self.yA)**2)
        #distance from current point P to final point B
        self.rho = np.sqrt((self.xB-self.currentstate[0])**2 + (self.yB-self.currentstate[1])**2)
        
        
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
        
        reward = 1-k1*(self.rho/self.rhoAB + k1*k2*abs(self.theta) + k3*(1-self.rho/self.rhoAB)*abs(self.theta))
        #reward normalisation according to the maximum number of steps
        reward = reward/self.config["MAX_STEPS"]
        return reward
    
    
    #check if done
    def isdone(self):
        done = False
        #done if the final point is almost reached (error due to discretisation is considered)
        #or if point B x coordinate is almost reached
        if abs(1 - abs((self.rhoAB - self.rho)/self.rhoAB)) <= 10**(-3) or abs(self.xB-self.currentstate[0]) <= 5*10**(-4):
            done = True
        return done
    
    #reset the model with initial conditions
    def reset(self):        
        self.currentstate = [self.xA, self.yA, self.uA, self.vA]
        return self.currentstate
        