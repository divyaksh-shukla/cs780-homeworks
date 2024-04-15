# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # ASSIGNMENT 4
# # Submission Deadline: April 16, 6PM
# # Submission Link: https://forms.gle/G4B6FiAsyoPLCkZu9  

# # Table of Contents
#
# 1. [Provide Information](#Provide-Information)
# 2. [Instructions](#Instructions)
# 3. [Environment](#Environment)
# 4. [Hyperparameters](#Hyperparameters)
# 5. [Helper Functions](#helper)
# 6. [DDPG](#ddpg)
# 7. [TD3](#td3)
# 8. [PPO](#ppo)
# 9. [Experiments to Run](#experiments)

# # Provide Information
# <a id="Provide-Information"></a>

# Name: **DIVYAKSH SHUKLA**
#
# Roll No.: **231110603**
#
# IITK EMail: **divyakshs23@iitk.ac.in**

# # Instructions
# <a id="Instructions"></a>
#

# **Read all the instructions below carefully before you start working on the assignment.**
# - The purpose of this course is that you learn RL and the best way to do that is by implementation and experimentation.
# - The assignment requires your to implement some algorithms and you are required report your findings after experimenting with those algorithms.
# - **You are required to submit ZIP file containing a Jupyter notebook (.ipynb), and an image folder. The notebook would include the code, graphs/plots of the experiments you run and your findings/observations. Image folder is the folder having plots, images, etc.**
# - In case you use any maths in your explanations, render it using latex in the Jupyter notebook.
# - You are expected to implement algorithms on your own and not copy it from other sources/class mates. Of course, you can refer to lecture slides.
# - If you use any reference or material (including code), please cite the source, else it will be considered plagiarism. But referring to other sources that directly solve the problems given in the assignment is not allowed. There is a limit to which you can refer to outside material.
# - This is an individual assignment.
# - In case your solution is found to have an overlap with solution by someone else (including external sources), all the parties involved will get zero in this and all future assignments plus further more penalties in the overall grade. We will check not just for lexical but also semantic overlap. Same applies for the code as well. Even an iota of cheating would NOT be tolerated. If you cheat one line or cheat one page the penalty would be same.
# - Be a smart agent, think long term, if you cheat we will discover it somehow, the price you would be paying is not worth it.
# - In case you are struggling with the assignment, seek help from TAs. Cheating is not an option! I respect honesty and would be lenient if you are not able to solve some questions due to difficulty in understanding. Remember we are there to help you out, seek help if something is difficult to understand.
# - The deadline for the submission is given above. Submit at least 30 minutes before the deadline, lot can happen at the last moment, your internet can fail, there can be a power failure, you can be abducted by aliens, etc.
# - You have to submit your assignment via the Google Form (link above)
# - The form would close after the deadline and we will not accept any solution. No reason what-so-ever would be accepted for not being able to submit before the deadline.
# - Since the assignment involves experimentation, reporting your results and observations, there is a lot of scope for creativity and innovation and presenting new perspectives. Such efforts would be highly appreciated and accordingly well rewarded. Be an exploratory agent!
# - Your code should be very well documented, there are marks for that.
# - In your plots, have a clear legend and clear lines, etc. Of course you would generating the plots in your code but you must also put these plots in your notebook. Generate high resolution pdf/svg version of the plots so that it doesn't pixilate on zooming.
# - For all experiments, report about the seed used in the code documentation, write about the seed used.
# - In your notebook write about all things that are not obvious from the code e.g., if you have made any assumptions, references/sources, running time, etc.
# -  **DO NOT Forget to write name, roll no and email details above**
# - **In addition to checking your code, we will be conducting one-on-one viva for the evaluation. So please make sure that you do not cheat!**
# - **Use of LLMs based tools or AI-based code tools is strictly prohibited! Use of ChatGPT, VS Code, Gemini, CO-Pilot, etc. is not allowed. NOTE VS code is also not allowed. Even in Colab disable the AI assistant. If you use it, we will know it very easily. Use of any of the tools would be counted as cheating and would be given a ZERO, with no questions asked.**
# - For each of the sub-part in the question create a new cell below the question and put your answer in there. This includes the plots as well

# # OpenAI Gym Environments
# <a id="Environment"></a>

# +
# all imports go in here
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import random
from itertools import count, cycle
from collections import deque, namedtuple
import os
import sys
from tqdm import tqdm, trange

import socket

HOSTNAME = socket.gethostname()
CWD = os.getcwd()
IMAGES_DIR = os.path.join(CWD, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
# -

# In this assignment we will be exploring Deep RL algorithms and for this we will be using environmentd provided by OpenAI Gym. In particular we will be exploring "Pendulum-v1" , "Hopper-v4", and "Half-Cheetah" environments (https://gymnasium.farama.org/environments/classic_control/ ). The code to instantiate the environments are given in the cells below. Run these cells and play with the environments to learn more details about the environments. 

# #### Pendulum-v1

# +
# # Create Inverted Pendulum environment
# #https://gymnasium.farama.org/environments/classic_control/cart_pole/

# N_EPISODES = 1
# MAX_EPISODE_STEPS = 100
# SEED = 34
# env = gym.make(
#     'Pendulum-v1', 
#     render_mode="rgb_array", 
#     max_episode_steps=MAX_EPISODE_STEPS
#     )
# s = env.reset(seed = SEED)
# print("Observation Space = ")
# print(env.observation_space)
# print(env.observation_space.shape)
# print("Action Space = ")
# print(env.action_space)
# print(env.action_space.shape)
# for episode in range(N_EPISODES):
#     print("In episode {}".format(episode))
    
#     s, _ = env.reset()
#     terminated = False
#     truncated = False
#     steps = 0
#     while not (truncated or terminated):
#         env.render()
#         print(s)
#         a = env.action_space.sample()
#         s, r, terminated, truncated, info = env.step(a)
#         steps += 1
#         if terminated or truncated:
#             print("Finished after {} timestep".format(steps))
#             break
# env.close()
# -

# #### Hopper-v4

# +
# if HOSTNAME != 'divyaksh-hp': # Installing packages in colab or kaggle. But not on local machine
# #     !pip install gymnasium
# #     !pip install swig
# #     !pip install gymnasium[box2d]
# #     !pip install gymnasium[mujoco]

# +
# # Create Hopper environment
# # https://gymnasium.farama.org/environments/mujoco/hopper/

# N_EPISODES = 1
# MAX_EPISODE_STEPS = 100
# SEED = 34

# import gymnasium as gym
# env = gym.make("Hopper-v4", render_mode = "rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
# s = env.reset(seed = SEED)
# print("Observation Space = ")
# print(env.observation_space)
# print("Action Space = ")
# print(env.action_space)
# terminated = False
# for episode in range(N_EPISODES):
#     print("In episode {}".format(episode))
#     s, _ = env.reset()
#     terminated = False
#     truncated = False
#     steps = 0
#     while not (terminated or truncated):
#         env.render()
#         print(s)
#         a = env.action_space.sample()
#         s, r, terminated, truncated, _ = env.step(a)
#         steps += 1
#         if terminated or truncated:
#             print("Finished after {} timestep".format(steps))
# env.close()

# -

# #### HalfCheetah-v4

# +
# # Create Half-Cheetah environment
# # https://gymnasium.farama.org/environments/mujoco/hopper/

# N_EPISODES = 1
# MAX_EPISODE_STEPS = 100
# SEED = 34

# import gymnasium as gym
# env = gym.make("HalfCheetah-v4", render_mode = "rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
# s = env.reset(seed = SEED)
# print("Observation Space = ")
# print(env.observation_space)
# print("Action Space = ")
# print(env.action_space)
# terminated = False
# for episode in range(N_EPISODES):
#     print("In episode {}".format(episode))
#     s, _ = env.reset()
#     terminated = False
#     truncated = False
#     steps = 0
#     while not (terminated or truncated):
#         env.render()
#         print(s)
#         a = env.action_space.sample()
#         s, r, terminated, truncated, _ = env.step(a)
#         steps += 1
#         if terminated or truncated:
#             print("Finished after {} timestep".format(steps))
# env.close()

# -

# # Hyperparameters
# <a id="Hyperparameters"></a>
#
# All your hyperparameters should be stated here. We will change their value here and your code should work  accordingly. 

# +
# mention the values of all the hyperparameters (you can add more hyper-paramters as well) to be used in the entire notebook, put the values that gave the best
# performance and were finally used for the agent
N_EPISODES = 1
SEED = 34
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 256
EPOCHS = 10
TAU = 0.005

MAX_TRAIN_EPISODES = 120
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -

# # Helper Functions
# <a id="helper"></a>
#
# Write all the helper functions that will be used for value-based and policy based algorithms below. In case you want to add more helper functions, please feel free to add.

# +
class ValueNetworkDDPG(nn.Module):
    def __init__(self, inDim, outDim, hDim = [32,32], action_dim = None, activation = F.relu, device=DEVICE):
        super(ValueNetworkDDPG, self).__init__()
        self.activation = activation
        self.inputLayer = nn.Linear(inDim, hDim[0])
        self.hiddenLayer = []
        for i in range(len(hDim)-1):
            self.hiddenLayer.append(nn.Linear(hDim[i] + action_dim, hDim[i+1] + action_dim, device=device))
        self.outputLayer = nn.Linear(hDim[-1] + action_dim, outDim)
        
    def forward(self, x, actions):
        x = self.activation(self.inputLayer(x))
        x = torch.cat([x, actions], 1)
        for layer in self.hiddenLayer:
            x = self.activation(layer(x))
        x = self.outputLayer(x)
        return x
    
class ValueNetworkTD3(nn.Module):
    def __init__(self, inDim, outDim, hDim = [32,32], action_dim = None, activation = F.relu, device=DEVICE):
        super(ValueNetworkTD3, self).__init__()
        self.activation = activation
        self.inputLayer = nn.Linear(inDim + action_dim, hDim[0])
        self.hiddenLayer = []
        for i in range(len(hDim)-1):
            self.hiddenLayer.append(nn.Linear(hDim[i], hDim[i+1], device=device))
        self.outputLayer = nn.Linear(hDim[-1], outDim)
        
    def forward(self, x, actions, actionRange):
        actionLow, actionHigh = actionRange
        actions = (actions - actions.min()) * (actionHigh - actionLow)/(actions.max() - actions.min()) + actionLow
        
        x = torch.cat([x, actions], 1)
        
        x1 = self.activation(self.inputLayer(x))
        for layer in self.hiddenLayer:
            x1 = self.activation(layer(x1))
        x1 = self.outputLayer(x1)
        
        x2 = self.activation(self.inputLayer(x))
        for layer in self.hiddenLayer:
            x2 = self.activation(layer(x2))
        x2 = self.outputLayer(x2)
        
        return x1, x2

def createValueNetwork(inDim, outDim, hDim = [8, 8], action_dim = None, activation = F.relu, agent = None):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return q-value for each possible action
    assert agent in ['DDPG', 'TD3'], "Agent should be either DDPG or TD3"
    
    if agent == 'DDPG':
        valueNetwork = ValueNetworkDDPG(inDim=inDim, outDim=outDim, hDim=hDim, action_dim = action_dim, activation=activation, device=DEVICE).to(DEVICE)
    elif agent == 'TD3':
        valueNetwork = ValueNetworkTD3(inDim=inDim, outDim=outDim, hDim=hDim, action_dim = action_dim, activation=activation, device=DEVICE).to(DEVICE)
    
    
    return valueNetwork



# +
#Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, inDim, outDim, hDim = [32,32], activation = F.relu, device=DEVICE):
        super(PolicyNetwork, self).__init__()
        self.activation = activation
        self.inputLayer = nn.Linear(inDim, hDim[0])
        self.hiddenLayer = []
        for i in range(len(hDim)-1):
            self.hiddenLayer.append(nn.Linear(hDim[i], hDim[i+1], device=device))
        self.outputLayer = nn.Linear(hDim[-1], outDim)
        self.device = device
    
    def forward(self, x):
        x = self.activation(self.inputLayer(x))
        for layer in self.hiddenLayer:
            x = self.activation(layer(x))
        x = self.outputLayer(x)
        return x

def createPolicyNetwork(inDim, outDim, hDim = [32,32], activation = F.relu):
    #this creates a Feed Forward Neural Network class and instantiates it and returns the class
    #the class should be derived from torch nn.Module and it should have init and forward method at the very least
    #the forward function should return action logit vector 
    #Your code goes in here
    
    policyNetwork = PolicyNetwork(inDim, outDim, hDim, activation, device=DEVICE).to(DEVICE)
    
    return policyNetwork


# -

# ## ReplayBuffer 

# In next few cells, you will implement replaybuffer class. 
#
# This class creates a buffer for storing and retrieving experiences. This is a generic class and can be used
# for different agents like NFQ, DQN, DDQN, PER_DDQN, etc. 
# Following are the methods for this class which are implemented in subsequent cells
#
# ```
# class ReplayBuffer():
#     def __init__(self, bufferSize, batch_size, seed)
#     def store(self, state, action, reward, next_state, done)
#     def sample(self, batchSize)
#     def length(self)
# ```   

from collections import deque


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, seed):
        # this function creates the relevant data-structures, and intializes all relevant variables
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.isPriorityBuffer = False
        self.cumulative_reward = 0
        self.episode_steps = 0
        self.seed = seed
        random.seed(seed)


class ReplayBuffer(ReplayBuffer):
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


class ReplayBuffer(ReplayBuffer):
    def sample(self, batchSize):
        # this method returns batchSize number of experiences
        # this function returns experiences samples
        #
        
        experiencesList = random.sample(self.buffer, batchSize)
        
        return experiencesList


class ReplayBuffer(ReplayBuffer):
    def splitExperiences(self, experiences, device):
        # this method returns batchSize number of experiences
        # this function returns experiences samples
        
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones


class ReplayBuffer(ReplayBuffer):
    def length(self):
        #tells the number of experiences stored in the internal buffer
        
        return len(self.buffer)


# # Deep Deterministic Policy Gradient (DDPG)
# <a id="ddpg"></a>

# Implement the Deep Deterministic Policy Gradient (DDPG) agent. We have studied about DDPG agent in the Lecture. Use the function definitions (given below).
#
# This class implements the DDPG agent, you are required to implement the various methods of this class
# as outlined below. Note this class is generic and should work with any permissible Gym environment
#
# ```
# class DDPG():
#     def init(self, env, seed, gamma, tau, bufferSize, batch_size, updateFrequency,
#              policyOptimizerFn, valueOptimizerFn,
#              policyOptimizerLR,valueOptimizerLR,
#              MAX_TRAIN_EPISODES,MAX_EVAL_EPISODE,
#              optimizerFn)
#     
#     def runDDPG(self)
#     def trainAgent(self)
#     def gaussianStrategy(self, net , s , envActionRange , noiseScaleRatio,
#         explorationMax = True)
#     def greedyStrategy(self, net , s , envActionRange)
#     def trainNetworks(self, experiences)
#     def updateNetworks(self, onlineNet, targetNet, tau)
#     def evaluateAgent(self)
#
#
#
#
# ```

class DDPG():
    def __init__(self, env, seed, gamma, tau, buffer_size, batch_size, update_frequency, epochs, 
             valueHdim, policyHdim,
             policyOptimizerFn, valueOptimizerFn,
             policyOptimizerLR, valueOptimizerLR,
             MAX_TRAIN_EPISODES,MAX_EVAL_EPISODE = 1,
             optimizerFn = None):
        #this DDPG method 
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc. 
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates targetValueNetwork , targetPolicyNetwork
        # 4. creates and initializes (with network params) the optimizer function
        # 5. creates onlineValueNetwork, onlinePolicyNetwork 
        # 6. Creates the replayBuffer
        
        self.env = env
        self.actionDim = env.action_space.shape[0]
        self.stateDim = env.observation_space.shape[0]
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.epochs = epochs
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODE = MAX_EVAL_EPISODE
        
        self.noiseScaleRatio = NOISE_SCALE_RATIO
        
        self.targetValueNetwork = createValueNetwork(self.stateDim, 1, hDim=valueHdim, action_dim=self.actionDim, agent="DDPG")
        self.onlineValueNetwork = createValueNetwork(self.stateDim, 1, hDim=valueHdim,action_dim=self.actionDim, agent="DDPG")
        self.valueOptimizer = valueOptimizerFn(self.onlineValueNetwork.parameters(), lr=valueOptimizerLR)
        
        self.targetPolicyNetwork = createPolicyNetwork(self.stateDim, self.actionDim, hDim=policyHdim, activation=lambda x: torch.tanh(x)*2)
        self.onlinePolicyNetwork = createPolicyNetwork(self.stateDim, self.actionDim, hDim=policyHdim, activation=lambda x: torch.tanh(x)*2)
        self.policyOptimizer = policyOptimizerFn(self.onlinePolicyNetwork.parameters(), lr=policyOptimizerLR)
        
        self.replayBuffer = ReplayBuffer(buffer_size, batch_size, seed)
    


class DDPG(DDPG):
    def updateNetworks(self):
        #this function updates the onlineNetwork with the target network
        #
        mixed_weights = zip(self.onlineValueNetwork.parameters(), self.targetValueNetwork.parameters())
        self.targetValueNetwork.load_state_dict({key: self.tau*o + (1-self.tau)*t for key, (o, t) in zip(self.targetValueNetwork.state_dict().keys(), mixed_weights)})
        
        mixed_weights = zip(self.onlinePolicyNetwork.parameters(), self.targetPolicyNetwork.parameters())
        self.targetPolicyNetwork.load_state_dict({key: self.tau*o + (1-self.tau)*t for key, (o, t) in zip(self.targetPolicyNetwork.state_dict().keys(), mixed_weights)})



class DDPG(DDPG):
    def gaussianStrategy (self, net , s , envActionRange , noiseScaleRatio ,
        explorationMax = True ):
        #this function sets the scale of exploration then add the noise of this scale to the greedy action
        #and clips it within the range
        
        actionLowVal, actionHighVal = self.env.action_space.low, self.env.action_space.high
        
        if explorationMax:
            noiseScale = actionHighVal
        else:
            noiseScale = noiseScaleRatio * actionHighVal
        state = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        greedyAction = net(state).detach().cpu().numpy()
        noise = np.random.normal(0, noiseScale, self.actionDim)
        action = np.clip(greedyAction + noise, actionLowVal, actionHighVal)
        return action



class DDPG(DDPG):
    def greedyStrategy (self, net , s , envActionRange):
        #this function selects the greedy action
        #and clips it within the range
        
        actionLowVal, actionHighVal = self.env.action_space.low, self.env.action_space.high
        
        state = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        greedyAction = net(state).detach().cpu().numpy()
        action = np.clip(greedyAction, actionLowVal, actionHighVal)
        return action



class DDPG(DDPG):
    def runDDPG (self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards 
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode 
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training, 
        #                               note this will include time for BookKeeping and evaluation 
        # Note both trainTime and wallClockTime get accumulated as episodes proceed. 
        #
        trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, episodeSteps = self.trainAgent()
        # resultEval = self.evaluateAgent()
        
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, evalRewardsList[-1], episodeSteps



class DDPG(DDPG):
    def trainAgent(self):
        #this method collects experiences and trains the agent and does BookKeeping while training. 
        #this calls the trainNetwork() method internally, it also evaluates the agent per episode
        #it trains the agent for MAX_TRAIN_EPISODES
        #
        
        self.updateNetworks()
        s, _ = self.env.reset(seed = self.seed)
        
        # Book-keeping
        trainRewardsList = np.zeros(self.MAX_TRAIN_EPISODES)
        evalRewardsList = np.zeros(self.MAX_TRAIN_EPISODES)
        trainTimeList = np.zeros(self.MAX_TRAIN_EPISODES)
        wallClockTimeList = np.zeros(self.MAX_TRAIN_EPISODES)
        episodeSteps = np.zeros(self.MAX_TRAIN_EPISODES)
        
        looper = trange(MAX_TRAIN_EPISODES, ncols=120, desc='DDPG Training')
        
        for e in looper:
            episodeStartTime = time.time()
            episodeTrainTime = 0
            episodeWallClockTime = 0
            s, _ = self.env.reset(seed = self.seed)
            terminated = False
            truncated = False
            steps = 0
            episodeReward = 0
            while not (terminated or truncated):
                a = self.gaussianStrategy(self.onlinePolicyNetwork, s, (self.env.action_space.low, self.env.action_space.high), self.noiseScaleRatio, self.replayBuffer.length() < MIN_SAMPLES)
                next_s, r, terminated, truncated, _ = self.env.step(a)
                self.replayBuffer.store(s, a, r, next_s, terminated or truncated)
                episodeReward += r
                steps += 1
                s = next_s
                if self.replayBuffer.length() > MIN_SAMPLES:
                    trainStartTime = time.time()
                    self.trainNetwork(self.replayBuffer.sample(self.batch_size))
                    episodeTrainTime += time.time() - trainStartTime
                if e % self.update_frequency == 0:
                    self.updateNetworks()
            
            # Book-keeping
            evalRewardsList[e] = self.evaluateAgent()
            trainRewardsList[e] = episodeReward
            trainTimeList[e] = episodeTrainTime
            episodeSteps[e] = steps
            episodeWallClockTime = time.time() - episodeStartTime
            wallClockTimeList[e] = episodeWallClockTime
            looper.set_postfix(TrainReward=trainRewardsList[e], EvalReward=evalRewardsList[e], TotalSteps=episodeSteps[e])
            
        
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, episodeSteps


class DDPG(DDPG):
    def trainNetwork(self, experiences):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss 
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc. 
        #
        
        states, actions, rewards, next_states, dones = self.replayBuffer.splitExperiences(experiences, DEVICE)
        for epoch in range(self.epochs):
            argmax_a_qs_v = self.targetPolicyNetwork(next_states).detach()
            max_a_qs_v = self.targetValueNetwork(next_states, argmax_a_qs_v).detach()
            target_qs = rewards + self.gamma * max_a_qs_v * (1 - dones)
            predicted_qs = self.onlineValueNetwork(states, actions)
            loss = F.mse_loss(predicted_qs, target_qs)
            self.valueOptimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.onlineValueNetwork.parameters(), 1)
            self.valueOptimizer.step()
            
            predicted_actions = self.onlinePolicyNetwork(states)
            loss = -self.onlineValueNetwork(states, predicted_actions).mean()
            self.policyOptimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.onlinePolicyNetwork.parameters(), 1)
            self.policyOptimizer.step()


class DDPG(DDPG):
    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1
        #
        rewards = np.zeros(self.MAX_EVAL_EPISODE)
        for e in range(self.MAX_EVAL_EPISODE):
            reward = 0
            s, _ = self.env.reset(seed = self.seed)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                a = self.greedyStrategy(self.onlinePolicyNetwork, s, (self.env.action_space.low, self.env.action_space.high))
                s, r, terminated, truncated, _ = self.env.step(a)
                reward = r + self.gamma * reward
                steps += 1
            rewards[e] = reward
            
        
        return rewards.mean()


# # Twin-Delayed Deep Deterministic Policy Gradient (TD3) 
# <a id="td3"></a>

# Implement the Twin-delayed deep deterministic policy gradient (TD3) agent. We have studied about TD3 agent in the Lecture. Use the function definitions (given below).
#
# This class implements the TD3 agent, you are required to implement the various methods of this class
# as outlined below. Note this class is generic and should work with any permissible Gym environment
#
# ```
# class DDPG():
#     def init(env, gamma, tau,
#     bufferSize ,
#     updateFrequencyPolicy ,
#     updateFrequencyValue ,
#     trainPolicyFrequency ,
#     policyOptimizerFn ,
#     valueOptimizerFn ,
#     policyOptimizerLR ,
#     valueOptimizerLR ,
#     MAX TRAIN EPISODES,
#     MAX EVAL EPISODE,
#     optimizerFn )
#     
#     def runTD3 (self)
#     def trainAgent (self)
#     def gaussianStrategy (self, net , s , envActionRange , noiseScaleRatio ,
#         explorationMax = True)
#     def greedyStrategy (self, net , s , envActionRange)
#     def trainNetworks (self,experiences , envActionRange)
#     def updateValueNetwork(self, onlineNet, targetNet, tau)
#     def updatePolicyNetwork(self, onlineNet, targetNet, tau)
#     def evaluateAgent (self)
#
#
#
# ```

class TD3():
    def __init__(self,env, seed, gamma, tau,
    buffer_size , batch_size ,
    policy_update_frequency ,
    value_update_frequency ,
    policy_train_frequency , 
    epochs ,
    valueHdim , policyHdim ,
    policyOptimizerFn ,
    valueOptimizerFn ,
    policyOptimizerLR ,
    valueOptimizerLR ,
    MAX_TRAIN_EPISODES,
    MAX_EVAL_EPISODE,
    optimizerFn):
        #this TD3 method 
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc. 
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates targetValueNetwork , targetPolicyNetwork
        # 4. creates and initializes (with network params) the optimizer function
        # 5. creates onlineValueNetwork, onlinePolicyNetwork 
        # 6. Creates the replayBuffer
        
        self.env = env
        self.actionDim = env.action_space.shape[0]
        self.stateDim = env.observation_space.shape[0]
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.value_update_frequency = value_update_frequency
        self.policy_update_frequency = policy_update_frequency
        self.policy_train_frequency = policy_train_frequency
        self.epochs = epochs
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODE = MAX_EVAL_EPISODE
        
        self.noiseScaleRatio = NOISE_SCALE_RATIO
        
        self.targetValueNetwork = createValueNetwork(self.stateDim, 1, hDim=valueHdim, action_dim=self.actionDim, agent='TD3')
        self.onlineValueNetwork = createValueNetwork(self.stateDim, 1, hDim=valueHdim, action_dim=self.actionDim, agent='TD3')
        self.valueOptimizer = valueOptimizerFn(self.onlineValueNetwork.parameters(), lr=valueOptimizerLR)
        
        self.targetPolicyNetwork = createPolicyNetwork(self.stateDim, self.actionDim, hDim=policyHdim, activation=lambda x: torch.tanh(x)*2)
        self.onlinePolicyNetwork = createPolicyNetwork(self.stateDim, self.actionDim, hDim=policyHdim, activation=lambda x: torch.tanh(x)*2)
        self.policyOptimizer = policyOptimizerFn(self.onlinePolicyNetwork.parameters(), lr=policyOptimizerLR)
        
        self.replayBuffer = ReplayBuffer(buffer_size, batch_size, seed)
    


class TD3(TD3):
    def updateValueNetwork(self, tau):
        #this function updates the onlineValueNetwork with the targetValuenetwork
        #
        mixed_weights = zip(self.onlineValueNetwork.parameters(), self.targetValueNetwork.parameters())
        self.targetValueNetwork.load_state_dict({key: self.tau*o + (1-self.tau)*t for key, (o, t) in zip(self.targetValueNetwork.state_dict().keys(), mixed_weights)})



class TD3(TD3):
    def updatePolicyNetwork(self, tau):
        #this function updates the onlinePolicuNetwork with the targetPolicynetwork
        #
        mixed_weights = zip(self.onlinePolicyNetwork.parameters(), self.targetPolicyNetwork.parameters())
        self.targetPolicyNetwork.load_state_dict({key: self.tau*o + (1-self.tau)*t for key, (o, t) in zip(self.targetPolicyNetwork.state_dict().keys(), mixed_weights)})



class TD3(TD3):
    def gaussianStrategy (self, net , s , envActionRange , noiseScaleRatio ,
        explorationMax = True ):
        #this function sets the scale of exploration then add the noise of this scale to the greedy action
        #and clips it within the range
        
        actionLowVal, actionHighVal = self.env.action_space.low, self.env.action_space.high
        
        if explorationMax:
            noiseScale = actionHighVal
        else:
            noiseScale = noiseScaleRatio * actionHighVal
        state = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        greedyAction = net(state).detach().cpu().numpy()
        noise = np.random.normal(0, noiseScale, self.actionDim)
        action = np.clip(greedyAction + noise, actionLowVal, actionHighVal)
        return action



class TD3(TD3):
    def greedyStrategy (self, net , s , envActionRange ):
        #this function selects the greedy action
        #and clips it within the range

        actionLowVal, actionHighVal = self.env.action_space.low, self.env.action_space.high
        
        state = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        greedyAction = net(state).detach().cpu().numpy()
        action = np.clip(greedyAction, actionLowVal, actionHighVal)
        return action



class TD3(TD3):
    def runTD3 (self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards 
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode 
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training, 
        #                               note this will include time for BookKeeping and evaluation 
        # Note both trainTime and wallClockTime get accumulated as episodes proceed. 
        #
        trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, episodeSteps = self.trainAgent()
        # resultEval = self.evaluateAgent()
        
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, evalRewardsList[-1], episodeSteps



class TD3(TD3):
    def trainAgent(self):
        #this method collects experiences and trains the agent and does BookKeeping while training. 
        #this calls the trainNetwork() method internally, it also evaluates the agent per episode
        #it trains the agent for MAX_TRAIN_EPISODES
        #
        self.updateValueNetwork(self.tau)
        self.updatePolicyNetwork(self.tau)
        s, _ = self.env.reset(seed = self.seed)
        
        # Book-keeping
        trainRewardsList = np.zeros(self.MAX_TRAIN_EPISODES)
        evalRewardsList = np.zeros(self.MAX_TRAIN_EPISODES)
        trainTimeList = np.zeros(self.MAX_TRAIN_EPISODES)
        wallClockTimeList = np.zeros(self.MAX_TRAIN_EPISODES)
        episodeSteps = np.zeros(self.MAX_TRAIN_EPISODES)
        
        looper = trange(MAX_TRAIN_EPISODES, ncols=120, desc='TD3 Training')
        
        for e in looper:
            episodeStartTime = time.time()
            episodeTrainTime = 0
            episodeWallClockTime = 0
            s, _ = self.env.reset(seed = self.seed)
            terminated = False
            truncated = False
            steps = 0
            episodeReward = 0
            while not (terminated or truncated):
                a = self.gaussianStrategy(self.onlinePolicyNetwork, s, (self.env.action_space.low, self.env.action_space.high), self.noiseScaleRatio, self.replayBuffer.length() < MIN_SAMPLES)
                next_s, r, terminated, truncated, _ = self.env.step(a)
                self.replayBuffer.store(s, a, r, next_s, terminated or truncated)
                episodeReward = r + self.gamma * episodeReward
                steps += 1
                s = next_s
                if self.replayBuffer.length() > MIN_SAMPLES:
                    trainStartTime = time.time()
                    self.trainNetwork(self.replayBuffer.sample(self.batch_size), (self.env.action_space.low, self.env.action_space.high), e)
                    episodeTrainTime += time.time() - trainStartTime
                if e % self.policy_update_frequency == 0:
                    self.updatePolicyNetwork(self.tau)
                if e % self.value_update_frequency == 0:
                    self.updateValueNetwork(self.tau)
            
            # Book-keeping
            trainRewardsList[e] = episodeReward
            trainTimeList[e] = episodeTrainTime
            episodeWallClockTime = time.time() - episodeStartTime
            wallClockTimeList[e] = episodeWallClockTime
            evalRewardsList[e] = self.evaluateAgent()
            episodeSteps[e] = steps
            looper.set_postfix(trainReward=episodeReward, evalReward=evalRewardsList[e], episodeSteps=steps)
            
        
        return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, episodeSteps


class TD3(TD3):
    def trainNetwork(self,experiences , envActionRange, episode):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss 
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc. 
        #
        actionLowVal, actionHighVal = envActionRange
        actionLowVal = torch.tensor(actionLowVal, dtype=torch.float32, device=DEVICE)
        actionHighVal = torch.tensor(actionHighVal, dtype=torch.float32, device=DEVICE)
        
        states, actions, rewards, next_states, dones = self.replayBuffer.splitExperiences(experiences, device=DEVICE)
        
        for epoch in range(self.epochs):    
            action_noise = torch.tensor(actionHighVal - actionLowVal, dtype=torch.float32, device=DEVICE) * torch.normal(0, 1, size=actions.shape, device=DEVICE)
            action_noise = torch.clamp(action_noise, actionLowVal, actionHighVal)
            
            argmax_a_qs_v = self.targetPolicyNetwork(next_states).detach()
            noisy_argmax_a_qs_v = argmax_a_qs_v + action_noise
            noisy_argmax_a_qs_v = torch.clamp(noisy_argmax_a_qs_v, actionLowVal, actionHighVal)
            max_1_a_qs_v, max_2_a_qs_v = self.targetValueNetwork(next_states, noisy_argmax_a_qs_v, (actionLowVal, actionHighVal))
            max_1_a_qs_v = max_1_a_qs_v.detach()
            max_2_a_qs_v = max_2_a_qs_v.detach()
            max_a_qs_v = torch.min(max_1_a_qs_v, max_2_a_qs_v)
            target_qs = rewards + self.gamma * max_a_qs_v * (1 - dones)
                    
            predicted_qs_1, predicted_qs_2 = self.onlineValueNetwork(states, actions, (actionLowVal, actionHighVal))
            loss = F.mse_loss(predicted_qs_1, target_qs) + F.mse_loss(predicted_qs_2, target_qs)
            
            self.valueOptimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.onlineValueNetwork.parameters(), 1)
            self.valueOptimizer.step()
            
            if episode % self.policy_train_frequency == 0:
                predicted_actions = self.onlinePolicyNetwork(states)
                max_1_a_qs_p, max_2_a_qs_p = self.onlineValueNetwork(states, predicted_actions, (actionLowVal, actionHighVal))
                loss = -torch.cat((max_1_a_qs_p, max_2_a_qs_p), dim=0).mean(dim=0)
                self.policyOptimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.onlinePolicyNetwork.parameters(), 1)
                self.policyOptimizer.step()


class TD3(TD3):
    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1
        #
        rewards = np.zeros(self.MAX_EVAL_EPISODE)
        for e in range(self.MAX_EVAL_EPISODE):
            reward = 0
            s, _ = self.env.reset(seed = self.seed)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                a = self.greedyStrategy(self.onlinePolicyNetwork, s, (self.env.action_space.low, self.env.action_space.high))
                s, r, terminated, truncated, _ = self.env.step(a)
                reward = r + self.gamma * reward
                steps += 1
            rewards[e] = reward
            
        
        return rewards.mean()

# # PPO
# <a id="PPO"></a>

# PPO have quite a few key implementation details. 
# Please Refer: 
# "Proximal Policy Optimization Algorithms" [PPO](https://arxiv.org/abs/1707.06347) and 
# "Implementation Matters in Deep RL: A Case Study on PPO and TRPO" [Implementation Matters](https://openreview.net/forum?id=r1etN1rtPB)
#
# Lets finish things off with an easy implementation of PPO!
# A easy way to check you implementation details is running your implementation on some easier environment first and make sure it converges. Like "CartPole-v1" should converge to episodic return of 500 in around 300k steps.

# +
# #All imports here
# ## Feel free to add or remove

# import os
# import random
# import time

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions.categorical import Categorical

# +
# #Hyperparameters
# gym_id = "CartPole-v1"  #The id of the gym environment
# learning_rate = 
# seed = 1
# total_timesteps =  #The total timesteps of the experiments
# torch_deterministic = True   #If toggled, `torch.backends.cudnn.deterministic=False
# cuda = True

# num_envs = 4  #The number of parallel game environments (Yes PPO works with vectorized environments)
# num_steps = 128 #The number of steps to run in each environment per policy rollout
# anneal_lr = True #Toggle learning rate annealing for policy and value networks
# gae = True #Use GAE for advantage computation
# gamma = 
# gae_lambda =  #The lambda for the general advantage estimation
# num_minibatches = 4
# update_epochs =  #The K epochs to update the policy
# norm_adv = True  #Toggles advantages normalization
# clip_coef =  #The surrogate clipping coefficient (See what is recommended in the paper!)
# clip_vloss = True #Toggles whether or not to use a clipped loss for the value function, as per the paper
# ent_coef =  #Coefficient of the entropy
# vf_coef =  #Coefficient of the value function
# max_grad_norm = 0.5
# target_kl = None #The target KL divergence threshold


# batch_size = int(num_envs * num_steps)
# minibatch_size = int(batch_size // num_minibatches)


# +
# #PPO works with vectorized enviromnets lets make a function that returns a function that returns an environment.
# #Refer how to make vectorized environments in gymnasium
# def make_env(gym_id, seed):
#     #Your code here
#     pass
# -

# #We initialize the layers in PPO , refer paper.
# #Lets initialize the layers with this function
# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     #Initializes the weights and bias of the layers
#     #Your code here
#     return layer


# +
# #Lets make the Main agent class
# class Agent(nn.Module):
#     def __init__(self, envs):
#         super(Agent, self).__init__()
#         #self.critic = # Your code here (Critic Network) 
#         #(Returns a single value of the observation)
        
#         #self.actor = # Your code here (Actor Network) 
#         #(Returns the logits of the actions on the observations)


# +
# class Agent(Agent):
#         def get_value(self, x):
#             # Returns the value from the critic on the observation x
#             return 

# +
# class Agent(Agent):
#     def get_action_and_value(self, x, action=None):
#         #Returns 1.the action (sampled according to the logits), 
#         #2.log_prob of the action,
#         #3.Entropy,
#         #4.Value from the critic
        
#         #Your code here
#         return 


# +
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = torch_deterministic

# device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")


# +
# #Make the vectorized environments, use the helper function that we have declared above
# envs = # Your code here

# +
# agent = Agent(envs).to(device)
# optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=) #eps is not the default that pytorch uses

# # ALGO Logic: Storage setup
# obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
# actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
# logprobs = torch.zeros((num_steps, num_envs)).to(device)
# rewards = torch.zeros((num_steps, num_envs)).to(device)
# dones = torch.zeros((num_steps, num_envs)).to(device)
# values = torch.zeros((num_steps, num_envs)).to(device)


# +
# # Start the game
# global_step = 0
# start_time = time.time()
# next_obs, info = envs.reset()
# next_obs = torch.Tensor(next_obs).to(device)
# next_done = torch.zeros(num_envs).to(device)
# num_updates = total_timesteps // batch_size

# +
# #This is the main training loop where we collect the experience , 
# #calculate the advantages, ratio , the total loss and learn the policy

# for update in range(1, num_updates + 1):
    
#     # Annealing the rate if instructed to do so.
#     if anneal_lr:
#         # Your code here 
#         pass

#     for step in range(0, num_steps):
#         global_step += 1 * num_envs  # We are taking a step in each environment 
#         obs[step] = next_obs
#         dones[step] = next_done

#         # ALGO LOGIC: action logic
#         with torch.no_grad():
#             #Get the action , logprob , _ , value from the agent.
            
#             action, logprob, _, value = # Your code here

#             values[step] = value.flatten()
#         actions[step] = action
#         logprobs[step] = logprob

#         # TRY NOT TO MODIFY: execute the game and log data.
#         next_obs, reward, terminated,truncated, info = envs.step(action.cpu().numpy())
#         rewards[step] = torch.tensor(reward).to(device).view(-1)
#         next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)
        
#         for item in info:
#             if item == "final_info" and info[item][0] is not None:
#                 print(f"global_step={global_step}, episodic_return={info[item][0]['episode']['r']}")
#                 break

#     # bootstrap value if not done
#     with torch.no_grad():
#         next_value = agent.get_value(next_obs).reshape(1, -1)
#         if gae:
#             pass
#             # Your code here
            
#             #returns = advantages + values  (yes official implementation of ppo calculates it this way)
#         else:
            
#             # Your code here 
#             pass

#             #advantages = returns - values

#     # flatten the batch
#     b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
#     b_logprobs = logprobs.reshape(-1)
#     b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
#     b_advantages = advantages.reshape(-1)
#     b_returns = returns.reshape(-1)
#     b_values = values.reshape(-1)

#     # Optimizing the policy and value network
#     b_inds = np.arange(batch_size)
#     clipfracs = []
#     for epoch in range(update_epochs):
#         #Get a random sample of batch_size
#         np.random.shuffle(b_inds)
#         for start in range(0, batch_size, minibatch_size):
#             end = start + minibatch_size
#             mb_inds = b_inds[start:end]

#             #Your code here
#             #Calculate the ratio
#             _, newlogprob, entropy, newvalue = 
#             logratio =  
#             ratio = 
            
#             with torch.no_grad():
#                 # calculate approx_kl http://joschu.net/blog/kl-approx.html
#                 # Refer the blog for calculating kl in a simpler way
#                 old_approx_kl = 
#                 approx_kl = 
#                 clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

#             mb_advantages = b_advantages[mb_inds]
#             if norm_adv:
#                 mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

#             # Policy loss (Calculate the policy loss pg_loss)
#             # Your code here 


#             # Value loss v_loss
#             newvalue = newvalue.view(-1)
#             if clip_vloss:
#                 pass
#             else:
#                 pass

#             # Entropy loss 
#             entropy_loss = 

#             # Total loss
#             loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

#             optimizer.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
#             optimizer.step()

#         if target_kl is not None:
#             if approx_kl > target_kl:
#                 break

#     y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
#     var_y = np.var(y_true)
#     explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


# envs.close()
# -

# # Experiments and Plots
# <a id="experiments"></a>

# Run the DDPG, TD3, PPO on Pendulum, Hopper and Half Cheetah environment respectively.
#
# Plot the following for each of the environment separately. Note based on different hyper-parameters and strategies you use, you can have multiple plots for each of the below. 
#
# As you are aware from your past experience, single run of the agent over the environment results in plots that have lot of variance and look very noisy. One way to overcome this is to create several different instances of the environment using different seeds and then average out the results across these and plot these. For all the plots below, you this strategy. You need to run 5 different instances of the environment for each agent. As you have seen in the lecture slides, we plot the maximum and minimum values around the mean in the plots, so this gives us the shaded plot with the mean curve in the between. In this assignment, you are required to do the same. Generate plots with envelop between maximum and minimum value
# For each of the quantity of interest, plot each of the agent within the same plot using different colors for the envelop. Choose colors such that that there is clear contrast between the plots corresponding to different agents.
#
# 1. Plot mean train rewards vs episodes 
# 2. Plot mean evaluation rewards vs episodes 
# 3. Plot total steps vs episode
# 4. Plot train time vs episode
# 5. Plot wall clock time vs episode
# 6. Based on plots what are your observations about DDPG and TD3, compare the two algorithms.
# 7. What is the advatage of PPO over DDPG or TD3?

# ### Pendulum-v1

# +
N_RUNS = 5
N_EPISODES = 500
MAX_EPISODE_STEPS = 120
ENV_NAME = 'Pendulum-v1'

IMAGES_DIR = os.path.join(CWD, 'images', ENV_NAME, 'v_notebook')
os.makedirs(IMAGES_DIR, exist_ok=True)

HYPERPARAM_FILE = os.path.join(CWD, 'hyperparameters_ddpg_td3.csv')
if not(os.path.exists(HYPERPARAM_FILE)):
    hy_file = open(HYPERPARAM_FILE, 'w')
    hy_file.write("AGENT,N_RUNS,MAX_EPISODE_STEPS,env,SEED,N_EPISODES,GAMMA,NOISE_SCALE_RATIO,MIN_SAMPLES,EPOCHS,TAU,BATCH_SIZE,UPDATE_FREQUENCY,BUFFER_SIZE,VALUE_HDIM,POLICY_HDIM,POLICY_OPTIM,VALUE_OPTIM,POLICY_LR,VALUE_LR,MAX_TRAIN_EPISODES,MAX_EVAL_EPISODES,DEVICE\n")

env = gym.make(
    ENV_NAME, 
    render_mode="rgb_array", 
    max_episode_steps=MAX_EPISODE_STEPS
)

trainRewards = np.zeros((3, N_RUNS, N_EPISODES))
trainTime = np.zeros((3, N_RUNS, N_EPISODES))
evalRewards = np.zeros((3, N_RUNS, N_EPISODES))
wallClockTime = np.zeros((3, N_RUNS, N_EPISODES))
episodeSteps = np.zeros((3, N_RUNS, N_EPISODES))

SEED = 42
random.seed(SEED)
SEEDS = [random.randint(0, 1000) for _ in range(N_RUNS)]
print(f'SEEDS = {SEEDS}')
# -

# DDPG

# +
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 128
EPOCHS = 40
TAU = 0.001
BATCH_SIZE = 64
UPDATE_FREQUENCY = 20
BUFFER_SIZE = 1000000

VALUE_HDIM = [400, 300]
POLICY_HDIM = [400, 300]
POLICY_OPTIM = optim.RMSprop
VALUE_OPTIM = optim.RMSprop
POLICY_LR = 1e-4
VALUE_LR = 1e-3

MAX_TRAIN_EPISODES = N_EPISODES
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hy_file = open(HYPERPARAM_FILE, 'a')
hy_file.write(f"DDPG,{N_RUNS},{MAX_EPISODE_STEPS},{ENV_NAME},{SEED},{N_EPISODES},{GAMMA},{NOISE_SCALE_RATIO},{MIN_SAMPLES},{EPOCHS},{TAU},{BATCH_SIZE},{UPDATE_FREQUENCY},{BUFFER_SIZE},{VALUE_HDIM},{POLICY_HDIM},{POLICY_OPTIM},{VALUE_OPTIM},{POLICY_LR},{VALUE_LR},{MAX_TRAIN_EPISODES},{MAX_EVAL_EPISODES},{DEVICE}\n")
hy_file.close()



for run in range(N_RUNS):
    print("Run {}".format(run))
    seed = SEEDS[run]
    ddpg = DDPG(env, seed, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, UPDATE_FREQUENCY, EPOCHS, VALUE_HDIM, POLICY_HDIM, POLICY_OPTIM, VALUE_OPTIM, POLICY_LR, VALUE_LR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES)
    trainRewards[0, run, :], trainTime[0, run, :], evalRewards[0, run, :], wallClockTime[0, run, :], _, episodeSteps[0, run, :] = ddpg.runDDPG()
# -

# TD3

# +
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 128
EPOCHS = 40
TAU = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 1000000
POLICY_UPDATE_FREQUENCY = 50
VALUE_UPDATE_FREQUENCY = 50
POLICY_TRAIN_FREQUENCY = 2

VALUE_HDIM = [400, 300]
POLICY_HDIM = [400, 300]
POLICY_OPTIM = optim.RMSprop
VALUE_OPTIM = optim.RMSprop
POLICY_LR = 1e-4
VALUE_LR = 1e-3

MAX_TRAIN_EPISODES = N_EPISODES
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hy_file = open(HYPERPARAM_FILE, 'a')
hy_file.write(f"TD3,{N_RUNS},{MAX_EPISODE_STEPS},{ENV_NAME},{SEED},{N_EPISODES},{GAMMA},{NOISE_SCALE_RATIO},{MIN_SAMPLES},{EPOCHS},{TAU},{BATCH_SIZE},{UPDATE_FREQUENCY},{BUFFER_SIZE},{VALUE_HDIM},{POLICY_HDIM},{POLICY_OPTIM},{VALUE_OPTIM},{POLICY_LR},{VALUE_LR},{MAX_TRAIN_EPISODES},{MAX_EVAL_EPISODES},{DEVICE}\n")
hy_file.close()

for run in range(N_RUNS):
    print("Run {}".format(run))
    seed = SEEDS[run]
    ddpg = TD3(env, seed, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, POLICY_UPDATE_FREQUENCY, VALUE_UPDATE_FREQUENCY, POLICY_TRAIN_FREQUENCY, EPOCHS, VALUE_HDIM, POLICY_HDIM, POLICY_OPTIM, VALUE_OPTIM, POLICY_LR, VALUE_LR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, None)
    trainRewards[1, run, :], trainTime[1, run, :], evalRewards[1, run, :], wallClockTime[1, run, :], _, episodeSteps[1, run, :] = ddpg.runTD3()


# -

# Plotting results

def plotResults(trainRewards, evalRewards, trainTime, wallClockTime, episodeSteps, agents, env_name):
    
    episodes = np.arange(trainRewards.shape[-1])
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ALPHA = 0.4
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    
    for i, agent in enumerate(agents):
        ax[0, 0].plot(episodes, trainRewards[i].mean(axis=0), label=agent, color=colors[i])
        ax[0, 0].fill_between(episodes, trainRewards[i].max(axis=0), trainRewards[i].min(axis=0), color=colors[i], alpha=ALPHA)
        title = f'Train Rewards on {env_name}'
        ax[0, 0].set_title(title)
        ax[0, 0].set_xlabel('Episodes')
        ax[0, 0].set_ylabel('Rewards')
        ax[0, 0].legend()
        
        ax[0, 1].plot(episodes, evalRewards[i].mean(axis=0), label=agent, color=colors[i])
        ax[0, 1].fill_between(episodes, evalRewards[i].max(axis=0), evalRewards[i].min(axis=0), color=colors[i], alpha=ALPHA)
        title = f'Evaluation Rewards on {env_name}'
        ax[0, 1].set_title(title)
        ax[0, 1].set_xlabel('Episodes')
        ax[0, 1].set_ylabel('Rewards')
        ax[0, 1].legend()
        
        ax[1, 0].plot(episodes, trainTime[i].cumsum(axis=1).mean(axis=0), label=agent, color=colors[i])
        ax[1, 1].fill_between(episodes, trainTime[i].cumsum(axis=1).max(axis=0), trainTime[i].cumsum(axis=1).min(axis=0), color=colors[i], alpha=ALPHA)
        title = f'Train Time on {env_name}'
        ax[1, 0].set_title(title)
        ax[1, 0].set_xlabel('Episodes')
        ax[1, 0].set_ylabel('Cumulative Train Time (s)')
        ax[1, 0].legend()
        
        ax[1, 1].plot(episodes, wallClockTime[i].cumsum(axis=1).mean(axis=0), label=agent, color=colors[i])
        ax[1, 1].fill_between(episodes, wallClockTime[i].cumsum(axis=1).max(axis=0), wallClockTime[i].cumsum(axis=1).min(axis=0), color=colors[i], alpha=ALPHA)
        title = f'Wall-clock Time on {env_name}'
        ax[1, 1].set_title(title)
        ax[1, 1].set_xlabel('Episodes')
        ax[1, 1].set_ylabel('Cumulative Wallclock Time (s)')
        ax[1, 1].legend()
        
        ax[2, 0].plot(episodes, episodeSteps[i].cumsum(axis=1).mean(axis=0), label=agent, color=colors[i])
        ax[2, 0].fill_between(episodes, episodeSteps[i].cumsum(axis=1).max(axis=0), episodeSteps[i].cumsum(axis=1).min(axis=0), color=colors[i], alpha=ALPHA)
        title = f'Total Steps on {env_name}'
        ax[2, 0].set_title(title)
        ax[2, 0].set_xlabel('Episodes')
        ax[2, 0].set_ylabel('Cumulative Steps')
        ax[2, 0].legend()
    
    with open(os.path.join(IMAGES_DIR), f'{env_name}_train_rewards.npy', 'wb') as f:
        np.save(f, trainRewards)
    with open(os.path.join(IMAGES_DIR), f'{env_name}_eval_rewards.npy', 'wb') as f:
        np.save(f, evalRewards)
    with open(os.path.join(IMAGES_DIR), f'{env_name}_train_time.npy', 'wb') as f:
        np.save(f, trainTime)
    with open(os.path.join(IMAGES_DIR), f'{env_name}_wallclock_time.npy', 'wb') as f:
        np.save(f, wallClockTime)
    with open(os.path.join(IMAGES_DIR), f'{env_name}_episode_steps.npy', 'wb') as f:
        np.save(f, episodeSteps)
    plt.savefig(os.path.join(IMAGES_DIR, f'{env_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight')


plotResults(trainRewards, evalRewards, trainTime, wallClockTime, episodeSteps, ["DDPG", "TD3"], ENV_NAME)

# ## Hopper-v4

# +
N_RUNS = 5
N_EPISODES = 1500
MAX_EPISODE_STEPS = 1000
ENV_NAME = 'Hopper-v4'

IMAGES_DIR = os.path.join(CWD, 'images', ENV_NAME, 'v_notebook')
os.makedirs(IMAGES_DIR, exist_ok=True)

HYPERPARAM_FILE = os.path.join(CWD, 'hyperparameters_ddpg_td3.csv')
if not(os.path.exists(HYPERPARAM_FILE)):
    hy_file = open(HYPERPARAM_FILE, 'w')
    hy_file.write("AGENT,N_RUNS,MAX_EPISODE_STEPS,env,SEED,N_EPISODES,GAMMA,NOISE_SCALE_RATIO,MIN_SAMPLES,EPOCHS,TAU,BATCH_SIZE,UPDATE_FREQUENCY,BUFFER_SIZE,VALUE_HDIM,POLICY_HDIM,POLICY_OPTIM,VALUE_OPTIM,POLICY_LR,VALUE_LR,MAX_TRAIN_EPISODES,MAX_EVAL_EPISODES,DEVICE\n")

env = gym.make(
    ENV_NAME, 
    render_mode="rgb_array", 
    max_episode_steps=MAX_EPISODE_STEPS
)

trainRewards = np.zeros((3, N_RUNS, N_EPISODES))
trainTime = np.zeros((3, N_RUNS, N_EPISODES))
evalRewards = np.zeros((3, N_RUNS, N_EPISODES))
wallClockTime = np.zeros((3, N_RUNS, N_EPISODES))
episodeSteps = np.zeros((3, N_RUNS, N_EPISODES))

SEED = 42
random.seed(SEED)
SEEDS = [random.randint(0, 1000) for _ in range(N_RUNS)]
print(f'SEEDS = {SEEDS}')
# -

# DDPG

# +
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 128
EPOCHS = 40
TAU = 0.001
BATCH_SIZE = 64
UPDATE_FREQUENCY = 20
BUFFER_SIZE = 1000000

VALUE_HDIM = [400, 300]
POLICY_HDIM = [400, 300]
POLICY_OPTIM = optim.RMSprop
VALUE_OPTIM = optim.RMSprop
POLICY_LR = 1e-4
VALUE_LR = 1e-3

MAX_TRAIN_EPISODES = N_EPISODES
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hy_file = open(HYPERPARAM_FILE, 'a')
hy_file.write(f"DDPG,{N_RUNS},{MAX_EPISODE_STEPS},{ENV_NAME},{SEED},{N_EPISODES},{GAMMA},{NOISE_SCALE_RATIO},{MIN_SAMPLES},{EPOCHS},{TAU},{BATCH_SIZE},{UPDATE_FREQUENCY},{BUFFER_SIZE},{VALUE_HDIM},{POLICY_HDIM},{POLICY_OPTIM},{VALUE_OPTIM},{POLICY_LR},{VALUE_LR},{MAX_TRAIN_EPISODES},{MAX_EVAL_EPISODES},{DEVICE}\n")
hy_file.close()



for run in range(N_RUNS):
    print("Run {}".format(run))
    seed = SEEDS[run]
    ddpg = DDPG(env, seed, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, UPDATE_FREQUENCY, EPOCHS, VALUE_HDIM, POLICY_HDIM, POLICY_OPTIM, VALUE_OPTIM, POLICY_LR, VALUE_LR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES)
    trainRewards[0, run, :], trainTime[0, run, :], evalRewards[0, run, :], wallClockTime[0, run, :], _, episodeSteps[0, run, :] = ddpg.runDDPG()


# -

# TD3

# +
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 128
EPOCHS = 40
TAU = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 1000000
POLICY_UPDATE_FREQUENCY = 50
VALUE_UPDATE_FREQUENCY = 50
POLICY_TRAIN_FREQUENCY = 2

VALUE_HDIM = [400, 300]
POLICY_HDIM = [400, 300]
POLICY_OPTIM = optim.RMSprop
VALUE_OPTIM = optim.RMSprop
POLICY_LR = 1e-4
VALUE_LR = 1e-3

MAX_TRAIN_EPISODES = N_EPISODES
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hy_file = open(HYPERPARAM_FILE, 'a')
hy_file.write(f"TD3,{N_RUNS},{MAX_EPISODE_STEPS},{ENV_NAME},{SEED},{N_EPISODES},{GAMMA},{NOISE_SCALE_RATIO},{MIN_SAMPLES},{EPOCHS},{TAU},{BATCH_SIZE},{UPDATE_FREQUENCY},{BUFFER_SIZE},{VALUE_HDIM},{POLICY_HDIM},{POLICY_OPTIM},{VALUE_OPTIM},{POLICY_LR},{VALUE_LR},{MAX_TRAIN_EPISODES},{MAX_EVAL_EPISODES},{DEVICE}\n")
hy_file.close()

for run in range(N_RUNS):
    print("Run {}".format(run))
    seed = SEEDS[run]
    ddpg = TD3(env, seed, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, POLICY_UPDATE_FREQUENCY, VALUE_UPDATE_FREQUENCY, POLICY_TRAIN_FREQUENCY, EPOCHS, VALUE_HDIM, POLICY_HDIM, POLICY_OPTIM, VALUE_OPTIM, POLICY_LR, VALUE_LR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, None)
    trainRewards[1, run, :], trainTime[1, run, :], evalRewards[1, run, :], wallClockTime[1, run, :], _, episodeSteps[1, run, :] = ddpg.runTD3()


# -

# ### Plotting Results

plotResults(trainRewards, evalRewards, trainTime, wallClockTime, episodeSteps, ["DDPG", "TD3"], ENV_NAME)

# ## HalfCheetah-v4

# +
N_RUNS = 5
N_EPISODES = 1500
MAX_EPISODE_STEPS = 1000
ENV_NAME = 'HalfCheetah-v4'

IMAGES_DIR = os.path.join(CWD, 'images', ENV_NAME, 'v_notebook')
os.makedirs(IMAGES_DIR, exist_ok=True)

HYPERPARAM_FILE = os.path.join(CWD, 'hyperparameters_ddpg_td3.csv')
if not(os.path.exists(HYPERPARAM_FILE)):
    hy_file = open(HYPERPARAM_FILE, 'w')
    hy_file.write("AGENT,N_RUNS,MAX_EPISODE_STEPS,env,SEED,N_EPISODES,GAMMA,NOISE_SCALE_RATIO,MIN_SAMPLES,EPOCHS,TAU,BATCH_SIZE,UPDATE_FREQUENCY,BUFFER_SIZE,VALUE_HDIM,POLICY_HDIM,POLICY_OPTIM,VALUE_OPTIM,POLICY_LR,VALUE_LR,MAX_TRAIN_EPISODES,MAX_EVAL_EPISODES,DEVICE\n")

env = gym.make(
    ENV_NAME, 
    render_mode="rgb_array", 
    max_episode_steps=MAX_EPISODE_STEPS
)

trainRewards = np.zeros((3, N_RUNS, N_EPISODES))
trainTime = np.zeros((3, N_RUNS, N_EPISODES))
evalRewards = np.zeros((3, N_RUNS, N_EPISODES))
wallClockTime = np.zeros((3, N_RUNS, N_EPISODES))
episodeSteps = np.zeros((3, N_RUNS, N_EPISODES))

SEED = 42
random.seed(SEED)
SEEDS = [random.randint(0, 1000) for _ in range(N_RUNS)]
print(f'SEEDS = {SEEDS}')
# -

# DDPG

# +
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 128
EPOCHS = 40
TAU = 0.001
BATCH_SIZE = 64
UPDATE_FREQUENCY = 20
BUFFER_SIZE = 1000000

VALUE_HDIM = [400, 300]
POLICY_HDIM = [400, 300]
POLICY_OPTIM = optim.RMSprop
VALUE_OPTIM = optim.RMSprop
POLICY_LR = 1e-4
VALUE_LR = 1e-3

MAX_TRAIN_EPISODES = N_EPISODES
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hy_file = open(HYPERPARAM_FILE, 'a')
hy_file.write(f"DDPG,{N_RUNS},{MAX_EPISODE_STEPS},{ENV_NAME},{SEED},{N_EPISODES},{GAMMA},{NOISE_SCALE_RATIO},{MIN_SAMPLES},{EPOCHS},{TAU},{BATCH_SIZE},{UPDATE_FREQUENCY},{BUFFER_SIZE},{VALUE_HDIM},{POLICY_HDIM},{POLICY_OPTIM},{VALUE_OPTIM},{POLICY_LR},{VALUE_LR},{MAX_TRAIN_EPISODES},{MAX_EVAL_EPISODES},{DEVICE}\n")
hy_file.close()



for run in range(N_RUNS):
    print("Run {}".format(run))
    seed = SEEDS[run]
    ddpg = DDPG(env, seed, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, UPDATE_FREQUENCY, EPOCHS, VALUE_HDIM, POLICY_HDIM, POLICY_OPTIM, VALUE_OPTIM, POLICY_LR, VALUE_LR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES)
    trainRewards[0, run, :], trainTime[0, run, :], evalRewards[0, run, :], wallClockTime[0, run, :], _, episodeSteps[0, run, :] = ddpg.runDDPG()


# -

# TD3

# +
GAMMA = 0.99
NOISE_SCALE_RATIO = 0.1
MIN_SAMPLES = 128
EPOCHS = 40
TAU = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 1000000
POLICY_UPDATE_FREQUENCY = 50
VALUE_UPDATE_FREQUENCY = 50
POLICY_TRAIN_FREQUENCY = 2

VALUE_HDIM = [400, 300]
POLICY_HDIM = [400, 300]
POLICY_OPTIM = optim.RMSprop
VALUE_OPTIM = optim.RMSprop
POLICY_LR = 1e-4
VALUE_LR = 1e-3

MAX_TRAIN_EPISODES = N_EPISODES
MAX_EVAL_EPISODES = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hy_file = open(HYPERPARAM_FILE, 'a')
hy_file.write(f"TD3,{N_RUNS},{MAX_EPISODE_STEPS},{ENV_NAME},{SEED},{N_EPISODES},{GAMMA},{NOISE_SCALE_RATIO},{MIN_SAMPLES},{EPOCHS},{TAU},{BATCH_SIZE},{UPDATE_FREQUENCY},{BUFFER_SIZE},{VALUE_HDIM},{POLICY_HDIM},{POLICY_OPTIM},{VALUE_OPTIM},{POLICY_LR},{VALUE_LR},{MAX_TRAIN_EPISODES},{MAX_EVAL_EPISODES},{DEVICE}\n")
hy_file.close()

for run in range(N_RUNS):
    print("Run {}".format(run))
    seed = SEEDS[run]
    ddpg = TD3(env, seed, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, POLICY_UPDATE_FREQUENCY, VALUE_UPDATE_FREQUENCY, POLICY_TRAIN_FREQUENCY, EPOCHS, VALUE_HDIM, POLICY_HDIM, POLICY_OPTIM, VALUE_OPTIM, POLICY_LR, VALUE_LR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, None)
    trainRewards[1, run, :], trainTime[1, run, :], evalRewards[1, run, :], wallClockTime[1, run, :], _, episodeSteps[1, run, :] = ddpg.runTD3()


# -

# ### Plotting Results

plotResults(trainRewards, evalRewards, trainTime, wallClockTime, episodeSteps, ["DDPG", "TD3"], ENV_NAME)
