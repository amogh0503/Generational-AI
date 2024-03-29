# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        """
        fc = full connections (all the neurons on 1 layer will be 
        connected to all the second layer)
        """
        nb_hiddenLayer = 30
        self.fc1 = nn.Linear(input_size, nb_hiddenLayer)
        # will connect the hidden layer w/ the output layer
        self.fc2 = nn.Linear(nb_hiddenLayer, nb_action)
    
    # Forward propagation function
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
    # Implementing Experience Replay
    
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event) # event has last state, new state, last action and last reward
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) #grouping memory in batches of state ,action and reward
        # returns the batches (action, reward, state)
        return map(lambda x: Variable(torch.cat(x,0)), samples) # mapping sample to pytorch variable
    
# Implementing Deep Q Learning
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.learning_rate = 0.001
        self.reward_window= []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) # 1 Lakh transition memory to learn from
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #intitalize tensor with input_size having 5 quantities 
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True))*7) # Putting softmax to output of NN ; temperature = 7
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #getting max of next states
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)      #temporal difference loss
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
     
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])))
        
        # Performs the action (take  random samples on memory)
        action = self.select_action(new_state)
        
        # Now we need to train and AI learn from the information 
        # contained in this samples of 100 random transitions
        nb_transitions = 100
        if len(self.memory.memory) > nb_transitions:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(nb_transitions)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
         # Reward_window has a fixed size
        reward_window_size = 1000
        if len(self.reward_window) > reward_window_size:
            del self.reward_window[0]
        return action
    # Calculates the mean of our rewards
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    # Saving our neural network and the optimizer into a file to be able to use later
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        # look for the file
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            # We update our existing model/optimizer to the file that is being loaded
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
    
        
        
        
        
        
        
        
        
        
