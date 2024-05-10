# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:09:44 2023

@author: Adrian Menor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from environment.env_config import env_parameters as params

class MapGenerator:
    def __init__(self, map_size: tuple, num_agents: int,num_obstacles: int, max_obstacle_width: float, max_obstacle_length: float, num_targets: int, collision_radius: float, spawn_radius: float, random_map: bool):
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.max_obstacle_width = max_obstacle_width
        self.max_obstacle_length = max_obstacle_length
        self.num_targets = num_targets
        self.collision_radius = collision_radius
        self.spawn_radius = spawn_radius
        self.num_agents = num_agents
        self.random_map = random_map
        
    def generate_agent_positions(self):
        '''
        Generates robots' initial positions around the center of the map.
        Checks whether the robots are spread beyond collision_radius
        given the spawn_radius.
        
        Output: Array with (x,y) initial position of robots.
        '''
        
        # Map center coordinates
        center_x, center_y = self.map_size[0] / 2, self.map_size[1] / 2
        
        # Raising error flag
        if self.num_agents < 2:
            raise Exception("Please provide a number of agents equal or larger than 2.")

        delta_angle = np.random.uniform(0, 2*np.pi)

        # Generating positions
        positions = []
                                                                                                                   
        for i in range(self.num_agents):
            # Calculate angle to spread the robots
            angle = 2 * np.pi * i / self.num_agents

            # Calculate robot positions
            x = center_x + self.spawn_radius * np.cos(angle + delta_angle)
            y = center_y + self.spawn_radius * np.sin(angle + delta_angle)
            positions.append((x, y))
            
        # Check whether the robots are beyond the collision radius
        agent0 = np.array(positions[0])
        agent1 = np.array(positions[1])
        
        if np.linalg.norm(agent1 - agent0 ) <= self.collision_radius * 1.05:
            raise Exception("Agents are not sufficiently spread, please use a larger spawn radius")

        # Converting the robot positions to numpy format
        positions = np.asarray(positions)
        
        return positions
    
    def generate_target_positions(self):
        '''
        Generates targets' positions.
        
        Output: Array with (x,y) position of targets.
        '''
        
        # Map center coordinates
        center_x, center_y = self.map_size[0] / 2, self.map_size[1] / 2
        
        # Generating target positions
        target_positions = []
        
        for i in range(self.num_targets):
            x = np.random.uniform(0 + self.collision_radius, self.map_size[0] - self.collision_radius)
            y = np.random.uniform(0 + self.collision_radius, self.map_size[1] - self.collision_radius)
            target_positions.append((x, y))
                
        # Dummy target to prevent environment crash
        target_positions.append((-1e6, -1e6))
                
        return target_positions
    
    def generate_obstacle_positions(self):
        '''
        Generates obstacles' positions.
        
        Output: Arrays with (x,y) centers of obstacles, widths, and lengths.
        '''
        
        # Map center coordinates
        center_x, center_y = self.map_size[0] / 2, self.map_size[1] / 2
        
        # Generating obstacles' center positions, widths, and lengths
        obstacle_centers = []
        obstacle_widths = []
        obstacle_lengths = []
        
        # Random delta angle to generate initial obstacle positions
        delta_angle = np.random.uniform(0, 2*np.pi)

        for i in range(7):
            # Calculate angle to spread the obstacles
            angle = 2 * np.pi * i / 7

            # Calculate obstacles' center positions
            x = center_x + (7 + self.max_obstacle_width + self.spawn_radius + self.collision_radius) * np.cos(angle + delta_angle)
            y = center_y + (7 + self.max_obstacle_length + self.spawn_radius + self.collision_radius) * np.sin(angle + delta_angle)
            
            if x >= self.map_size[0] or y >= self.map_size[1]:
                raise Exception("Obstacles are being generated outside the map, please make map_size larger, or spawn_radius smaller")
            
            obstacle_centers.append((x, y))
                
            # Generating obstacles' width and length
            if self.random_map:
                width = np.random.uniform(2, self.max_obstacle_width)
                length = np.random.uniform(2, self.max_obstacle_length)
                
                obstacle_widths.append(width)
                obstacle_lengths.append(length)
            
            else:
                obstacle_widths.append(self.max_obstacle_width)
                obstacle_lengths.append(self.max_obstacle_length)
                
        for i in range(15):
            x = np.random.uniform(0, self.map_size[0])
            y = np.random.uniform(0, self.map_size[1])
            
            # Checking that the obstacles are not generated on top of agents
            if np.abs(x-center_x)>self.spawn_radius + self.collision_radius and np.abs(y-center_y)>self.spawn_radius + self.collision_radius:
                obstacle_centers.append((x,y))
                obstacle_widths.append(self.max_obstacle_width)
                obstacle_lengths.append(self.max_obstacle_length)       
            
        # Placing obstacles at map's edges
        for i in range(21):
            obstacle_centers.append((0+i*5,0))
            obstacle_widths.append(self.max_obstacle_width)
            obstacle_lengths.append(self.max_obstacle_length)
            
        for i in range(21):
            obstacle_centers.append((0+i*5,100))
            obstacle_widths.append(self.max_obstacle_width)
            obstacle_lengths.append(self.max_obstacle_length)
            
        for i in range(21):
            obstacle_centers.append((0,0+i*5))
            obstacle_widths.append(self.max_obstacle_width)
            obstacle_lengths.append(self.max_obstacle_length)
            
        for i in range(21):
            obstacle_centers.append((100,0+i*5))
            obstacle_widths.append(self.max_obstacle_width)
            obstacle_lengths.append(self.max_obstacle_length)
            
        # Modifying centers to be a numpy array
        obstacle_centers = np.asarray(obstacle_centers)
                
        return obstacle_centers, obstacle_widths, obstacle_lengths
    
    def generate_terrain(self):
        '''
        Generates full terrain. 
        
        Output: agent_positions, target_positions, obstacle_centers, obstacle_widths, obstacle_lengths, base_position
        '''
        
        # Generating full terrain specifications
        agent_positions = self.generate_agent_positions()
        target_positions = self.generate_target_positions()
        obstacle_centers, obstacle_widths, obstacle_lengths = self.generate_obstacle_positions()
        
        # Base coordinates
        center_x, center_y = self.map_size[0] / 2, self.map_size[1] / 2
        base_position = (center_x, center_y)
        
        return agent_positions, target_positions, obstacle_centers, obstacle_widths, obstacle_lengths, base_position
    
    def plot_terrain(self):
        '''
        Plots the generated terrain. For visual inpection only.
        '''
        
        # Generating terrain parameters
        agent_positions, target_positions, obstacle_centers, obstacle_widths, obstacle_lengths, base_position = self.generate_terrain()
        map_size = self.map_size
        collision_radius = self.collision_radius
        
        
        fig, ax = plt.subplots()
    
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
    
        # Add legend
        legend_elements = [patches.Patch(facecolor='gray', edgecolor='black', label='Obstacles'),
                          plt.Line2D([0], [0], color='r', marker='x', linestyle='', markersize=8, label='Targets'),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Agents'),
                          plt.Line2D([0], [0], color='w', marker='s', markerfacecolor='blue', markersize=8, label='Base')]
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))  
        plt.show()
   
if __name__ == '__main__':
    # Example usage
    plt.close('all')
    terrain = MapGenerator(params["map_size"], params["num_agents"], params["num_obstacles"], params["max_obstacle_width"], params["max_obstacle_length"], params["num_targets"], params["collision_radius"], params["spawn_radius"], params["random_map"])
    agent_positions = terrain.generate_agent_positions()
    target_positions = terrain.generate_target_positions()
    obstacle_centers, obstacle_widths, obstacle_lengths = terrain.generate_obstacle_positions()
    terrain.plot_terrain()

