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

   1. please not use conda environment due to [ a known render issue](https://github.com/moderngl/moderngl/issues/469) 
   2. please use Python 3.7.6

### Basic install (if you just want to use existing environments without changing them)

```shell
pip3 install --upgrade pip
pip3 install git+https://github.com/dongfangliu/gym-fish.git
```

   ### Full installation (to edit/create environments) using a python virtual environment

```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
git clone https://github.com/dongfangliu/gym-fish.git
cd gym-fish
pip3 install -e .
```

## Getting Started

TO BE CONTINUED
