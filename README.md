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

2. Python Environment Setup
```
conda create -n gym_fish python=3.6.9
conda activate gym_fish
conda install -c conda-forge moderngl
```
3. Install Fish Gym
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

Now that we will try to run a basic task environment: point to point navigation.
```
import gym
import gym_fish
env = gym.make('fish-v0')

action = env.action_space.sample()
obs,reward,done,info = env.step(action)
env.render(mode = 'human')
```
