# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:17:59 2019 by Andrea
Last modified dec 17 2019 by Sandrine

"""

import numpy as np
from scipy.integrate import odeint


class FlatPlateModel:
    
    def __init__(self,initialconditions,finalposition,config):
        
        self.config = config
        
        self.xA = initialconditions[0]
        self.yA = initialconditions[1]
        self.A = np.array([self.xA,self.yA])
        self.uA = initialconditions[2]
        self.vA = initialconditions[3]
        self.xB = finalposition[0]
        self.yB = finalposition[1]
        self.B = np.array([self.xB,self.yB])
        self.rhoAB = np.linalg.norm(self.A-self.B)

        self.BA = self.A-self.B
        #print('self.BA', self.BA)
        self.phiA = np.arctan2(self.BA[1],self.BA[0])
        #print('self.phiA', self.phiA)

        #self.cartesian_state = np.array([self.xA,self.yA,self.uA,self.vA])
        #self.polar_state = self.get_state_in_relative_polar_coordinates()
        
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
        

    def get_state_in_relative_polar_coordinates(self, cartesian_state):
        BP = cartesian_state[0:2]-self.B
        rho = np.linalg.norm(BP)
        phiP = np.arctan2(BP[1],BP[0])
        theta = phiP - self.phiA

        #print('phiA', self.phiA)
        #print('phiP', phiP)
        #print('theta', theta)

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

        #print('phiA', self.phiA)
        #print('phiP', phiP)
        #print('theta', theta)

        x = self.xB + rho * np.cos(phiP)
        y = self.yB + rho * np.sin(phiP)

        u = rhoDot * np.cos(phiP) - thetaDot * np.sin(phiP)
        v = rhoDot * np.sin(phiP) + thetaDot * np.cos(phiP)
        cartesian_state = np.array([x, y, u, v])

        return cartesian_state


    # differential equations system for flat plate
    def flatplate(self,polar_state,t):
    #def flatplate(self):
        cartesian_state = self.get_state_in_absolute_cartesian_coordinates(polar_state)
        u = cartesian_state[2]
        v = cartesian_state[3]
        
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
        
        computedstate = np.array([dxdt, dydt, dudt, dvdt])
        return computedstate

    
    #compute next state, reward and done or not
    def step(self,action, polar_state):
        #alpha is the action
        self.alpha = action
        old_polar_state = polar_state
        #if some noise is wanted to be added to 'model' turbulence effects
        self.alpha = action + np.random.normal(scale=self.config["ACTION_SIGMA"])
        
        #time vector with initial and final point of the step
        timearray=np.linspace(0,self.config["DELTA_TIME"],2)
        old_cartesian_state = self.get_state_in_absolute_cartesian_coordinates(old_polar_state)
        #y = odeint(model, y0, t)
        odestates = odeint(self.flatplate,old_cartesian_state,timearray)
        #as odeint will return two different states, choose the second one
        new_cartesian_state = odestates[-1]
        new_polar_state = self.get_state_in_relative_polar_coordinates(new_cartesian_state)
        
        #compute reward and done
        reward = self.compute_reward(old_polar_state, action, new_polar_state)
        done = self.isdone(new_polar_state)
        
        # must return new state, reward, done (boolean)
        return [new_polar_state, reward, done]


    #compute reward
    def compute_reward(self, old_polar_state, action, new_polar_state):
        delta_rho = new_polar_state[0] - old_polar_state[0]
        delta_abs_theta = np.abs(new_polar_state[1]) - np.abs(old_polar_state[1])
        reward = -delta_rho # go to goal
        #reward = -delta_rho - delta_abs_theta # go to goal along the AB line

        return reward

    
    #check if done
    def isdone(self, polar_state):
        done = False
        #done if the final point is almost reached
        #or if abs(theta) >= pi/2
        #print('theta', polar_state[1])
        #print('np.abs(polar_state[1]+self.phiA)', np.abs(polar_state[1]+self.phiA))
        if np.abs(polar_state[0]/self.rhoAB) <= 10**(-3) or np.abs(polar_state[1]+self.phiA) >= np.pi/2.:
            done = True
        return done

    
    #reset the model with initial conditions
    def reset(self, state=None):
        if state==None:
            print('state none')
            cartesian_state = np.array([self.xA,self.yA,self.uA,self.vA])
            polar_state = self.get_state_in_relative_polar_coordinates(cartesian_state)
        else:
            print('state else')
            polar_state = state
        return polar_state
