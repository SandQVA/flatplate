# flatplate
simili gym environment to be used in conjunction with the SandQVA/cfd_sureli repository to perform a DRL of flying flat plate using a low-order model.

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
- run the command python3 train 'your_agent' --appli='flatplate'
- run the command python3 test 'your_agent' --appli='flatplate' --f='file created by the training phase'
