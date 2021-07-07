# Fish Gym
Fish Gym is a physics-based simulation framework for physical articulated underwater agent interaction with fluid.
This is the first physics-based environment that support coupled interation between agents and fluid in semi-realtime.
Fish Gym is integrated into the OpenAI Gym interface, enabling the use of existing reinforcement learning and control algorithms to control underwater agents to accomplish specific underwater exploration task.

Note this environment is now tested on Ubuntu18.04 and later, not tested on Mac, definitely not work on Windows due to dependency issues.

## Install
1. Install dependencies
  The only necessary dependency is [DART](https://github.com/dartsim/dart) and [CUDA](https://developer.nvidia.com/cuda-toolkit).
  For DART installation, please refer to [Here](https://dartsim.github.io/install_dart_on_ubuntu.html).
  We highly recommend to use latest version of DART and CUDA, specifically, DART 6.10.1 and CUDA 11.0 .
### Note for headless machines (server/cluster)
Please install dependencies for running a virtual display for scene rendering
```
sudo apt-install python3-pip mesa-utils libegl1-mesa xvfb
```

2. Python Environment Setup
```
conda create -n gym_fish python=3.6.9
conda activate gym_fish
conda install -c conda-forge moderngl
```
3. Install Fish Gym
```
git clone https://github.com/dongfangliu/gym-fish.git
cd gym-fish
pip3 install -e .
```
4.To test the learning compability, you may consider install stable-baselines3
```
conda activate gym_fish
pip3 install `stable-baselines3[extra]`
```


## Getting Started

### Run the environment
Now that we will try to run a basic task environment: point to point navigation.
```
import gym
import gym_fish
# Our environment runs on GPU to accelerate simulations,ensure a cuda-supported GPU exists on your machine
gpuId = 0

env = gym.make('fish-v0',gpuId =gpuId)

action = env.action_space.sample()
obs,reward,done,info = env.step(action)
```
### Render the scene
Then we can see the scene in two modes : `human` and `rgbarray`.
`human` is suitable for machines that with a display.
`rgbarray` is for headless machines.
#### For machines with a display
```
env.render(mode='human')
```
#### For headless machines (server/cluster)
Run a virtual display. for more details,check [here](https://moderngl.readthedocs.io/en/latest/techniques/headless_ubuntu_18_server.html)
```
export DISPLAY=:99.0
Xvfb :99 -screen 0 640x480x24 &
```
Render and outputs a numpy array
```
# This outputs a numpy array which can be saved as image
arr = env.render(mode='rgbarray`)
# Save use Pillow
from PIL import Image
image = Image.fromarray(arr)
# Then a scene output image can be viewed in 
image.save('output.png')
```
## Load and run a trained policy

## Train your own policy

## Evalute your policy

