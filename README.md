# flatplate
simili gym environment to be used in conjunction with the SandQVA/cfd_sureli repository to do DRL of a flying flat plate.

Requirements:
- Python 3.7 or older versions
- torch, torchvision
- imageio
- gym
- matplotlib
- PyYAML
- numpy

Usage : 
- Clone the SandQVA/cfd_sureli repository
- Inside this repository create a cfd folder
- Go to this new folder and clone SandQVA/flatplate repository
- Set the config.yaml file contained in cfd_sureli/cfd/flatplate/ according to the case you want to run
- run the command python3 train 'your_agent'
- run the command python3 test 'your_agent' --f='file created by the training phase'
