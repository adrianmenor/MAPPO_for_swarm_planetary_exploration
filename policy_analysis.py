# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:11:28 2024

@author: Adrian Menor
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time
import torch

# Importing environment classes
from environment.env_config import env_parameters
from environment.env_main import Environment
from environment.perception import calculate_perception as perception
from environment.perception import LIDAR as LIDAR

# Importing ANNs
from networks import ActorNetwork, CriticNetwork

# GIF maker
def make_gif(frame_folder, steps):
    # List files in the folder based on naming convention
    frames = [Image.open(os.path.join(frame_folder, f"step_{i}.png")) for i in range(steps)]

    # Save GIF
    frame_one = frames[0]
    frame_one.save("Saved_simulation.gif", format="GIF", append_images=frames,
                   save_all=True, duration=200, loop=0)
    
# Load the model
def load_model(model_class, model_name, input_dims, output_dims):
    # Get the current working directory
    current_directory = os.getcwd()

    # Specify the folder name
    folder_name = 'models'

    # Create the full path to the folder
    folder_path = os.path.join(current_directory, folder_name)

    # Create the full path to the file
    file_path = os.path.join(folder_path, f'{model_name}.pth')

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The model file {file_path} does not exist.")

    # Create an instance of the model class
    model = model_class(input_dims, output_dims)

    # Load the model state_dict
    model.load_state_dict(torch.load(file_path, map_location=model.device))

    return model

actor = load_model(ActorNetwork, 'actor_example', 108, 2)

# Create a folder to save images
os.makedirs("saved_images", exist_ok=True)

# Example usage
plotting = True
GIF = True # Needs plotting = True
steps = 100
env = Environment(env_parameters, plotting)
env.reset()
agent_positions = env.agent_positions 
target_positions = env.target_positions
obstacle_centers = env.obstacle_centers
obstacle_heights = env.obstacle_heights
obstacle_widths = env.obstacle_widths

initial_positions = np.reshape(agent_positions, (1, env.n_agents,2))

# Getting observations from the environment
# actor_obs, critic_obs = perception(agent_positions, target_positions, obstacle_centers, obstacle_heights, obstacle_widths)
lidar = LIDAR(agent_positions, target_positions, obstacle_centers, obstacle_heights, obstacle_widths)
actor_obs = lidar.FF_LIDAR()
critic_obs = actor_obs # CHANGE this

# Initialize an array to store agent positions at each timestep
trajectory_positions = np.zeros((steps, env.n_agents, 2))

initial_targets = target_positions.shape[0]

# Example episode
for steps in range(steps):    
    # Collectingactions from all agents
    actions = np.zeros((env.n_agents, 2))

    for agent in range(env.n_agents):
      action = actor.forward(actor_obs[:, agent])
      
      # Converting to NumPy
      action = action.detach().numpy()
      
      # Stacking actions to pass on 
      actions[agent, :] = action

    # Observing next state and reward
    new_agent_positions, rew, done, info = env.step(actions)
    # actor_obs, critic_obs = perception(new_agent_positions, info['target_positions'], info['obstacle_centers'], info['obstacle_heights'], info['obstacle_widths'])
    lidar = LIDAR(new_agent_positions, info['target_positions'], info['obstacle_centers'], info['obstacle_heights'], info['obstacle_widths'])
    actor_obs = lidar.FF_LIDAR()
    critic_obs = actor_obs # CHANGE thi
 
    # Store the agent positions at each timestep
    trajectory_positions[steps] = new_agent_positions
 
    if plotting:
        img_path = f"saved_images/step_{steps}.png"
        env.render()
        plt.savefig(img_path)

    if done:
        final_targets = env.agent_positions.shape[0]
        env.reset()

# Checking how many targets have been explored
final_targets = env.target_positions.shape[0]
print('Explored targets: ', initial_targets - final_targets)

# Making GIF
if GIF:
    make_gif("saved_images/", steps)

# Closing environment
env.close()

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

trajectory_positions = np.concatenate((initial_positions, trajectory_positions))

# Get the number of agents
num_agents = env.n_agents

# Define a colormap
color_map = plt.cm.get_cmap('tab20', num_agents)

for agent in range(num_agents):
    x = trajectory_positions[:, agent, 0]
    y = trajectory_positions[:, agent, 1]
    alphas = np.flip(1 - np.linspace(0, 1, len(x)))
    color = color_map(agent / num_agents * 1)  # Get color from the colormap
    axs[0].plot(x, y, alpha=0.5, color=color, linewidth=2.0)
    for i in range(len(x) - 1):
        axs[0].plot(x[i:i+2], y[i:i+2], alpha=alphas[i], color=color)
        
    # Plot a small circle at the beginning of the trajectory
    axs[0].scatter(x[0], y[0], color=color, marker='^', edgecolor='black', s=50, zorder=5)  # Adjust size as needed
    
    # Plot a slightly bigger circle at the end of the trajectory
    axs[0].scatter(x[-1], y[-1], color=color, edgecolor='black', s=100, zorder=5)  # Adjust size as needed

# Plot obstacles
for i in range(len(info['obstacle_centers'])):
    center = info['obstacle_centers'][i]
    width, height = info['obstacle_widths'][i], info['obstacle_heights'][i]
    rect = patches.Rectangle(center - np.array([width / 2, height / 2]), width, height, linewidth=1, edgecolor='black', facecolor='gray')
    axs[0].add_patch(rect)
    
for i in range(len(info['obstacle_centers'])):
    center = info['obstacle_centers'][i]
    width, height = info['obstacle_widths'][i], info['obstacle_heights'][i]
    rect = patches.Rectangle(center - np.array([width / 2, height / 2]), width, height, linewidth=1, edgecolor='black', facecolor='gray')
    axs[1].add_patch(rect)    
    
    
# Plot targets
for target in target_positions:
    axs[1].plot(target[0], target[1], 'rx', markersize=8, alpha=0.90)

# Add an overall title to the figure
fig.suptitle("Trajectories of MAPPO, Using Mixed Rewards, and a Rich Environment", fontsize=20, y=0.95)

# Set x and y axis limits for both subplots
for ax in axs:
    ax.set_xlim(0, env_parameters["map_size"][0])
    ax.set_ylim(0, env_parameters["map_size"][1])
    ax.set_aspect('equal')

# plt.savefig("trajectory_with_obstacles.pdf")
plt.show()