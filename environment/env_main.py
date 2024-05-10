# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:05:39 2023

@author: Adrian Menor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Importing environment classes
from environment.env_config import env_parameters
from environment.terrain import MapGenerator
from environment.physics import Dynamics
from environment.collision_detection import CollisionDetection
from environment.target_detection import check_targets

class Environment():
    def __init__(self, env_parameters: dict, Plotting=False):
        # Environment parameters
        self.params = env_parameters
        self.n_agents = env_parameters['num_agents']
        
        # Checking whether the environment will be rendered in the future
        if Plotting:
            # Initialize figure and axes
            self.fig, self.ax = plt.subplots()
            self.reset()
            self.initialize_plot()
            
        # Timestep counter
        self.time = 0

    def step(self, action):
        '''
        Updates the agents' positions based on the actions taken, also checking for collisions
        between obstacles.
        '''
        # Updating robot positions
        new_positions = self.robots.update_positions(self.agent_positions, action)
        self.agent_positions = new_positions
        
        # Calculating reward
        self.reward = self.reward_function()
        
        # Termination condition (when all targets are found)
        if self.target_positions.shape[0] - 1 > 0:
            done = False
        
        else:
            done = True
        
        # Environment information
        info = {"target_positions":self.target_positions, "obstacle_centers": self.obstacle_centers,
                "obstacle_widths": self.obstacle_widths, "obstacle_heights": self.obstacle_heights, 
                "base_position": self.base_position}
        
        # Updating time
        self.time += 1
        
        return self.agent_positions, self.reward, done, info

    def reset(self):
        '''
        Regenerates the environment terrain to start a new episode.
        '''
        # Generating terrain
        terrain = MapGenerator(self.params["map_size"], self.params["num_agents"], self.params["num_obstacles"], 
                               self.params["max_obstacle_width"], self.params["max_obstacle_length"], self.params["num_targets"], 
                               self.params["collision_radius"], self.params["spawn_radius"], self.params["random_map"])
        
        self.agent_positions, self.target_positions, self.obstacle_centers, self.obstacle_widths, self.obstacle_heights, self.base_position = terrain.generate_terrain()
        
        # Creating robots' dynamics object with current parameters
        self.robots = Dynamics(self.params["map_size"], self.obstacle_centers, self.obstacle_widths, self.obstacle_heights)
        
        # Creating robots' collision checker
        self.collisions = CollisionDetection(self.params["map_size"], self.params["collision_radius"], self.obstacle_centers, self.obstacle_widths, self.obstacle_heights)
        
        # Initializing reward
        self.reward = self.reward_function()
        
        return
    
    def reward_function(self):
        '''
        Calculates the reward of the current step. All robots get the same reward.
        '''
        # Checking collisions
        # collisions = self.collisions.check_collisions(self.agent_positions).count(True)
        individual_collisions = self.collisions.check_collisions(self.agent_positions)
        collisions = self.collisions.check_collisions(self.agent_positions).count(True)
        
        # Collision penalty
        # collision_penalty = -collisions / self.n_agents
        
        # Checking for found targets
        found_targets, self.target_positions, individual_found_targets = check_targets(self.agent_positions, 
                                                                                       self.params["perception_radius"], self.target_positions)
        
        # Curriculum learning; mixed ratio rewards
        reward = - np.asarray(individual_collisions, np.dtype(int)) / 7 + np.asarray(individual_found_targets) + 0.2 * found_targets * np.ones(self.params["num_agents"])
        
        # Curriculum learning; changing collision penalty severity
        # if self.time <= 1_000_000:
        #     penalty_factor = 1
            
        # else:        
        #     penalty_factor = np.clip((self.n_agents/(1e6))*(self.time-1e6), 1, self.n_agents)
        
        # Reward
        # reward = collision_penalty * penalty_factor + found_targets
        # reward = collision_penalty * self.n_agents + found_targets
        
        return reward

    def initialize_plot(self):
        '''
        Initializing plot.
        '''
        # Obtaining terrain parameters
        agent_positions = self.agent_positions
        target_positions = self.target_positions
        obstacle_centers = self.obstacle_centers
        obstacle_widths = self.obstacle_widths
        obstacle_lengths = self.obstacle_heights
        base_position = self.base_position
        map_size = self.params["map_size"]
        collision_radius = self.params["collision_radius"]

        # Initializing plot
        _, ax = self.fig, self.ax
    
        # Plot obstacles
        for i in range(len(obstacle_centers)):
            center = obstacle_centers[i]
            width, height = obstacle_widths[i], obstacle_lengths[i]
            rect = patches.Rectangle(center - np.array([width / 2, height / 2]), width, height, linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)
    
        # Plot targets
        for target in target_positions:
            ax.plot(target[0], target[1], 'rx', markersize=8)
    
        # Plot agents
        for agent in agent_positions:
            circle = plt.Circle((agent[0], agent[1]), collision_radius, color='blue', alpha=0.2)
            ax.add_patch(circle)
            ax.scatter(agent[0], agent[1], color='blue', marker='o')
    
        # Plot base position
        ax.plot(base_position[0], base_position[1], 'bs', markersize=8)
    
        # Set plot limits
        ax.set_xlim(0, map_size[0])
        ax.set_ylim(0, map_size[1])
        ax.set_aspect('equal', adjustable='box')
        
        # Calculate explored targets
        explored_targets = self.params["num_targets"] - len(self.target_positions) + 1
        
        # Define the text to display
        explored_text = f'Explored Targets: {explored_targets}'
        reward_text = f'Reward: {self.reward.mean()}'
        
        # Concatenate both pieces of text
        combined_text = f'{explored_text}\n{reward_text}'
        
        # Adding text box in the top-right
        text_x, text_y = map_size[0] * 0.9, map_size[1] * 0.9
        ax.text(text_x, text_y, combined_text, fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black'))

        # Add legend
        legend_elements = [patches.Patch(facecolor='gray', edgecolor='black', label='Obstacles'),
                          plt.Line2D([0], [0], color='r', marker='x', linestyle='', markersize=8, label='Targets'),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Agents'),
                          plt.Line2D([0], [0], color='w', marker='s', markerfacecolor='blue', markersize=8, label='Base')]
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))  
        plt.ion()
        plt.show()

    def render(self):
        '''
        Visualizes current environment state.
        '''
        # Clearing the axes to update the plot
        self.ax.clear()
        
        # Re-initializing the plot elements
        self.initialize_plot()  
        
        # Plotting
        plt.draw()
        plt.pause(0.001)

    def close(self):
        # Cleaning up resources
        plt.close('all')
        return
    
if __name__ == '__main__':
    # Example usage
    plotting = False
    env = Environment(env_parameters, plotting)
    env.reset()
    
    # Example episode
    for steps in range(1_000):
        actions =  env.action_space.sample() #+ np.array([-1, 1])
        new_positions, reward, done, info = env.step(actions)
        print(steps)
        if plotting:
            env.render()
        
        if done:
            env.reset()
        
    # Closing environment
    env.close()      