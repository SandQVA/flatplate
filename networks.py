# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:30:14 2019

@author: andrea
"""

import torch
import torch.optim as optim
import torch.nn as nn



class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)
    
    def save(self, file):
        torch.save(self.state_dict(), file)
        
    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return torch.tanh(self.output(x))
    
    def save(self, file):
        torch.save(self.state_dict(), file)
        
    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))
    


class Critic:
    def __init__(self, state_size, action_size, device, config):
        self.device = device

        self.nn = CriticNetwork(state_size + action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn = CriticNetwork(state_size + action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config["LEARNING_RATE_CRITIC"])

    def update(self, loss, grad_clipping=False):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self, tau):
        for target_param, nn_param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.sate.copy_((1-tau)*target_param.data + tau*nn_param.data)

    def save(self, folder):
        self.nn.save(folder+'/models/critic.pth')
        self.target_nn.save(folder+'/models/critic_target.pth')

    def load(self, folder):
        self.nn.load(folder+'/models/critic.pth', device=self.device)
        self.target_nn.load(folder+'/models/critic_target.pth', device=self.device)

    def target(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.target_nn(state_action)

    def __call__(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.nn(state_action)


class Actor:
    def __init__(self, state_size, action_size, device, config):
        self.device = device

        self.nn = ActorNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn = ActorNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config["LEARNING_RATE_ACTOR"])

    def update(self, loss, grad_clipping=False):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def update_target(self,tau):
        for target_param, nn_param in zip(self.target_nn.parameter(), self.nn.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*nn_param.data)

    def save(self, folder):
        self.nn.save(folder+'/models/actor.pth')
        self.target_nn.save(folder+'/models/actor_target.pth')

    def load(self, folder):
        self.nn.load(folder+'/models/actor.pth', device=self.device)
        self.target_nn.load(folder+'/models/actor_target.pth', device=self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().numpy()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)
