# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:17:11 2023

@author: Adrian Menor
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class Dynamics:
    def __init__(self, map_size: tuple, obstacle_centers: np.array, obstacle_widths: list, obstacle_heights: list)-> np.array:
        self.max_x, self.max_y = map_size
        self.obstacle_centers = obstacle_centers
        self.obstacle_widths = obstacle_widths
        self.obstacle_heights = obstacle_heights

    def update_positions(self, robot_positions, actions)-> np.array:
        '''
        Updates the position of the robots. If a collision occurs, the previous
        position is maintained.
        
        Dynamics are as follows:
        x <= x + Δx
        y <= y + Δy
        '''
        self.robot_positions = robot_positions
        new_positions = self.robot_positions + np.clip(actions,-2,2) # Actions are clipped
        final_positions = np.zeros((new_positions.shape[0], 2))
        for i, (new_x, new_y) in enumerate(new_positions):
            # Check if the new position collides with an obstacle or map boundary
            for j, (obs_x, obs_y) in enumerate(self.obstacle_centers):
                width = self.obstacle_widths[j]
                height = self.obstacle_heights[j]
                if (
                    (new_x >= obs_x - width / 2 and new_x <= obs_x + width / 2)
                    and (new_y >= obs_y - height / 2 and new_y <= obs_y + height / 2)
                ) or new_x < 0 or new_x > self.max_x or new_y < 0 or new_y > self.max_y:
                    new_positions[i] = self.robot_positions[i]
                    break
            final_positions[i] = np.array([new_positions[i, 0], new_positions[i, 1]])

        return final_positions

    def plot_positions(self):
        plt.figure(figsize=(6, 6))
        for position in self.robot_positions:
            plt.plot(position[0], position[1], marker='o', color='b')
        for i, (obs_x, obs_y) in enumerate(self.obstacle_centers):
            width = self.obstacle_widths[i]
            height = self.obstacle_heights[i]
            plt.gca().add_patch(Rectangle((obs_x - width / 2, obs_y - height / 2), width, height, linewidth=1, edgecolor='r', facecolor='none'))
        plt.xlim(0, self.max_x)
        plt.ylim(0, self.max_y)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Robots and Obstacles Plot')
        plt.show()

if __name__ == '__main__':
    from IPython.display import clear_output
    
    # Example usage
    map_size = (10, 10)
    robot_positions = np.array([[3, 3], [1.1, 5], [2, 8], [0.1, 2]])  # Example positions
    obstacle_centers = np.array([[2, 2], [5, 5]])
    obstacle_widths = [2, 2]
    obstacle_heights = [2, 3]
    actions = np.array([[0, 1], [0, -0.2], [1, 0], [.2, 2]])  # Example actions
    
    robot = Dynamics(map_size, obstacle_centers, obstacle_widths, obstacle_heights)
    
    plt.close("all")
    for _ in range(10):  # Run for 10 iterations
        robot_positions = robot.update_positions(robot_positions, actions)
        robot.plot_positions()
        plt.show()
        time.sleep(0.1)
        clear_output(wait=True)




