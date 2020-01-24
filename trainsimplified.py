# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:21:33 2019

@author: andrea
"""
import os
import time
from datetime import datetime
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import numpy as np
import matplotlib.pyplot as plt
import yaml

import torch
from tensorboardX import SummaryWriter

from model import Model
from flatplate import FlatPlateModel


def train():
    
    print("DDPG starting...")
    
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    
    parser = argparse.ArgumentParser(description='Run DDPG on ' + config["MODEL"])
    parser.add_argument('--no-gpu', action='store_true', dest='no_gpu', help="Don't use GPU")
    args = parser.parse_args()
    
    # Create folder and writer to write tensorboard values
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    
    if config["MODEL"] == "FlatPlate":
    
        #folder name
        folder = f'runsFlatPlate/{config["MODEL"].split("-")[0]}_{current_time}'
        writer = SummaryWriter(folder)
        if not os.path.exists(folder+'/models/'):
            os.mkdir(folder+'/models/')
        
        # Write optional info about the experiment to tensorboard
        for k, v in config.items():
            writer.add_text('Configuration', str(k) + ' : ' + str(v), 0)
        
        with open(folder+'/configuration.yaml', 'w') as file:
            yaml.dump(config, file)
        
        # Choose device cpu or cuda if a gpu is available
        if not args.no_gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
        writer.add_text('Device', device, 0)
        device = torch.device(device)
        
        
        #----------------------------------- Environment settigs ------------------------------
        
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
        
        low_bound_random = -12*np.pi/180
        high_bound_random = 12*np.pi/180
        
        ## --------------------------------------------------------
        
        env = FlatPlateModel(initialconditions,finalstate,config)
        

    
    print("Creating neural networks and optimizers...")
    #DDPG model generation
    model = Model(device, STATE_SIZE, ACTION_SIZE, folder, config)
    
    nb_total_steps = 0
    time_beginning = time.time()
    
    try:
        print("Starting training...")
        nb_episodes = 1 
        rewards = []                                #rewards array to be filled with episodes rewards
        
        columnepisode = 1                           #parameter to classify the information 
                                                    #of each episode in matrix form
        
        #evaluation of episodes
        for episode in trange(config["MAX_EPISODES"]):
            #print('episode', columnepisode)
            
            rowstep = 0                             #parameter to classify the information 
                                                    #of each episode in matrix form
            done = False                            #not finished the evaluation
            step = 0                                #start the step counter each new episode
            episode_reward = 0                      #initialise the reward for each new episode
            
            state = env.reset()                     #reset the model to start a new sequence
            
            #evaluation of steps within the current episode
            print('Episode number',nb_episodes,'keep waiting...')
            #check that not finished the episode
            while not done and step < config["MAX_STEPS"]:
                
                rowstep += 1                        #parameter to classify the information 
                                                    #of each episode in matrix form
                
                #action selection
                if nb_total_steps < config["RANDOM_EPISODES"]*config["MAX_STEPS"]:
                    #select at the begining of the evaluation with random actions
                    action = np.random.uniform(low_bound_random,high_bound_random,ACTION_SIZE)
                    if yB > 0:
                        action = abs(action)
                    elif yB < 0:
                        action = -abs(action)
                else:
                    action = model.select_action(state)
                #noise to help exploring new possibilities (at maximum, alpha=+-4degrees)
                noise = np.random.normal(scale=config["EXPLO_SIGMA"], size=ACTION_SIZE)
                action = np.clip(action+noise, LOW_BOUND, HIGH_BOUND)
                
                # Perform an action 
                next_state, reward, done = env.step(action)
                
                #check if there has been divergence in the solution
                if reward == 0 and np.array_equal(next_state,initialconditions):
                    reward = -config["MAX_STEPS"]
                    
                episode_reward += reward            #sum of all rewards until the final point
                
                #set as done if the number of maximum steps is reached even if not
                #reached the final position
                if not done and step == config["MAX_STEPS"] - 1:
                    done = True
                
                # Save transition into memory
                model.memory.push(state, action, reward, next_state, 1-int(done))
                state = next_state
                
                actor_loss, critic_loss = model.optimize()
                
                #increase the step number (per episode and total)
                step += 1
                nb_total_steps += 1
                
                #------------------------------- finish steps

            # SAVE variables at the end of episode
            env.fill_array_tobesaved()
            #increase the episode number    
            columnepisode += 1
            #add the total reward in the episode
            rewards.append(episode_reward)
            
            print('Episode reward:',episode_reward)
            
            
            writer.add_scalar('episode_rewards/actor', episode_reward, episode)
            if actor_loss is not None:
                writer.add_scalar('loss/actor_loss', actor_loss, episode)
            if critic_loss is not None:
                writer.add_scalar('loss/critic_loss', critic_loss, episode)
            writer.add_scalar('episode_rewards/actor', episode_reward, episode)
            
            if nb_episodes % config["FREQ_PLOT"] == 0:
                plt.cla()
                plt.plot(rewards)
                plt.grid()
                plt.title(folder[5:], fontsize = 18)
                plt.xlabel('Episodes', fontsize = 14)
                plt.ylabel('Cumulative reward', fontsize = 14)
                plt.xticks(fontsize=13)
                plt.yticks(fontsize=13)
                plt.savefig(folder+'/rewards.png')
            
            
            nb_episodes += 1
            #----------------------------------- finish episode
            
    except KeyboardInterrupt:
        pass
        
    finally:        
        
        # DUMP variables at the end of episode
        env.print_arrays_in_file(folder)

        writer.close()
        model.save()
        print("\033[91m\033[1mModel saved in", folder, "\033[0m")
        
        plt.show()
            
        print('Information saved in corresponding .csv files')
            
            #plot some paths (first random, first computed by AI, last path, ideal path)
          # plt.cla()
          # plt.plot(xmatrix[:,1],ymatrix[:,1],label='First random path')
          # plt.plot(xmatrix[:,config["RANDOM_EPISODES"]],ymatrix[:,config["RANDOM_EPISODES"]],label='First AI path')
          # plt.plot(xmatrix[:,nb_episodes-1],ymatrix[:,nb_episodes-1],label='Last path')
          # plt.plot([xA,xB],[yA,yB],label='Ideal path')
          # plt.grid()
          # plt.title('Trajectory modification', fontsize=18)
          # plt.xlabel('x position (m)', fontsize=14)
          # plt.ylabel('y position (m)', fontsize=14)
          # plt.xticks(fontsize=13)
          # plt.yticks(fontsize=13)
          # plt.legend(fontsize = 14)
          # plt.savefig(folder+'/trajectory.png')
        
        
    time_execution = time.time() - time_beginning
    #status printed
    print('---------------------------------------------------\n'
          '---------------------STATS-------------------------\n'
          '---------------------------------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes-1, ' episodes done\n'
          'Execution time ', round(time_execution, 2), ' seconds\n'
          'Execution time ', round(time_execution/60, 2), ' minutes\n'
          '---------------------------------------------------\n'
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
          'Average duration of one episode : ', round(time_execution/nb_episodes, 3), 's\n'
          '---------------------------------------------------')

#main of the code
train()
