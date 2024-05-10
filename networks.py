# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:52:45 2023

@author: Adrian Menor
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ActorNetwork, self).__init__()
        
        # Checking if a GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Creating ANN
        self.layer1 = nn.Linear(input_dims, 64, device=self.device)
        self.layer2 = nn.Linear(64, 64, device=self.device)
        self.layer3 = nn.Linear(64, output_dims, device=self.device)
        
    def forward(self, obs):
        ''' 
        Does a forward pass on the network. It maps S-> A.
        '''
        
        # Converting observation to PyTorch tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
            
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        pi = self.layer3(x)
        
        return pi
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        
        # Checking if a GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Creating ANN
        self.layer1 = nn.Linear(input_dims, 64, device=self.device)
        self.layer2 = nn.Linear(64, 64, device=self.device)
        self.layer3 = nn.Linear(64, 1, device=self.device) # Outputs Q-value
        
    def forward(self, obs):
        ''' 
        Does a forward pass on the network. It maps S-> Q.
        '''
        
        # Converting observation to PyTorch tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
            
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        q = self.layer3(x)
        
        return q
    
class CNN_CriticNetwork(nn.Module):
    def __init__(self, input_channels, grid_size):
        super(CNN_CriticNetwork, self).__init__()
        
        # ANN configuration parameters
        self.input_channels = input_channels
        self.grid_size = grid_size
        
        # Checking if a GPU is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=16, kernel_size=3, stride=1, padding=0,device=self.device)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0,device=self.device)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0,device=self.device)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0,device=self.device)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0,device=self.device)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2)
        
        # Define the fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 100,device=self.device)
        self.fc2 = nn.Linear(100, 30,device=self.device)
        self.fc3 = nn.Linear(30, 1,device=self.device)

    def forward(self, x):
        ''' 
        Does a forward pass on the network. It maps S-> Q.
        Input x: [batch_size, input_channels, grid_size, grid_size].
        '''
        # Converting observation to PyTorch tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float, device=self.device)
        
        # Apply convolution, activation, and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))

        # Flatten the input for the fully connected layers
        x = self.flatten(x)
        
        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        
        return q
    
    def print_parameters(self):
        total_params = 0
        for param_name, param in self.named_parameters():
            if param.requires_grad:
                num_params = np.prod(param.size())
                print(f"Parameter: {param_name}, Size: {param.size()}, Number of Parameters: {num_params}")
                total_params += num_params
        print(f"Total number of parameters: {total_params}")
    
if __name__ == "__main__":
    # Example usage
    actor = ActorNetwork(7, 2)
    critic = CriticNetwork(7)
    obs = np.ones((1,7)) # [Batch size, input dims]
    pi = actor.forward(obs)
    q = critic.forward(obs)
    print("Actor output:", pi)
    print("Critic output:", q)
    
    