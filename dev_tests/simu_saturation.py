import os
import yaml
import numpy as np
from flatplate import FlatPlateModel
import matplotlib.pyplot as plt

with open('configuration.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
state = env.reset()

states = []
rewards = []
action = .7*HIGH_BOUND
done = False
step = 0
print(config["MAX_STEPS"])
while not done and step < config["MAX_STEPS"]:
	if step%50 == 0:
		print(step)
	next_state, reward, done = env.step(action, state)
	state = next_state
	states.append(env.get_state_in_absolute_cartesian_coordinates(state))
	rewards.append(reward)
	step+=1

states = np.array(states)
plt.cla()
plt.plot([xA,xB],[yA,yB],label='Ideal path',c='b')
plt.plot(states[:,0],states[:,1],c='r')
plt.grid()
plt.xlabel('x position (m)', fontsize=14)
plt.ylabel('y position (m)', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize = 14)
plt.savefig('simu_saturation_traj.png')

plt.cla()
plt.plot(rewards)
plt.savefig('simu_saturation_reward.png')

print(np.sum(rewards))