# Deep Reinforcement Learning Nanodegree
## Project: Navigation
Andrzej Wodecki

December 25th, 2018



## Project details

The goal of the project is to **train the Agent to navigate and pick-up bananas** in the Banana environment provided by Unity Environments.

Agents receives +1 reward for collecting a yellow banana, and -1 for collecting a blue one. After training it should go for yellow bananas, and avoid blue ones.

**The state space** has 37 dimensions like the agent's velocity, along with ray-based perception of objects around agent's forward direction.

**The action space** consists of 4 discrete actions:

1. move forward: 0
2. move backward: 1
3. turn left: 3
4. turn right: 4

This is **episodic** environment. It is considered **solved** when agents gets an average score of +13 over 100 consecutive episodes.



## Getting started

First, you will need the Banana Environment provided by Unity - download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Place the file in the main folder of this repositore and unzip (or decompress) it.

You will also need a set of python packages installed, including jupyter, numpy and pytorch. All are provided within UDACITY "drlnd" environment: follow the instructions provided eg. [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Specifically:

1. create and activate a new environment with Python 3.6:
   `conda create --name drlnd python=3.6`
   `source activate drlnd`

2. In order to run the code provided in Navigation.ypnb jupyter notebook, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment:

   ```
   python -m ipykernel install --user --name drlnd --display-name "drlnd"
   ```


## Instructions

**

To run train the agent and see it action please open *Navigation.ypnb* notebook and follow the instructions.

Moreover, in the *grid_search* subfolder you will find a python code (*gridsearch.py*) used to grid search the Deep Q-Network hyperparameter space together with slightly adopted *dgn_agent.py* file. You can follow instructions within *gridsearch.py* to look for the optimal set of learning parameters: raw data from my research are stored in *1. large gamma tau.txt* and *2. buffor update.txt*. 

**IMPORTANT**: **Banana Environment is not provided in my repository.** To run the codes successfully you should install it (See *Getting Started* section) and modify (if necessary) first lines of *Navigation.ypnb* notebook and *gridsearch.py* pointing to the specific Banana environment, eg:

`#Instantiate the Environment and Agent`

`env = UnityEnvironment(file_name="Banana_linux/Banana.x86_64")`



# DeepRL-navigation
