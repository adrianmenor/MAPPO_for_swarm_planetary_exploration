# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:07:44 2023

@author: Adrian Menor
"""

import numpy as np
import matplotlib.pyplot as plt

class CollisionDetection:
    def __init__(self, map_size: tuple, collision_radius: float, obstacle_centers: np.array, obstacle_widths: list, obstacle_heights: list)-> list:
        '''
        Checks whether the robots have any kind of collisions.
        '''
        self.map_size = map_size 
        self.collision_radius = collision_radius
        self.obstacle_centers = obstacle_centers
        self.obstacle_widths = obstacle_widths
        self.obstacle_heights = obstacle_heights

    def check_agent_collisions(self, robot_positions):
        '''
        Checks for robot-robot collisions.
        
        Output: Array with collision booleans (True for collision, else False).
        '''
        n = len(robot_positions)
        collisions = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(robot_positions[i] - robot_positions[j]) <= self.collision_radius:
                    collisions[i] = True
                    collisions[j] = True
        return collisions

    def check_agent_obstacle_collisions(self, robot_positions):
        '''
        Checks for robot-obstacle collisions.
        
        Output: Array with collision booleans (True for collision, else False).
        '''
        n = len(robot_positions)
        m = len(self.obstacle_centers)
        collisions = [False] * n
        for i in range(n):
            for j in range(m):
                x_dist = np.abs(robot_positions[i][0] - self.obstacle_centers[j][0])
                y_dist = np.abs(robot_positions[i][1] - self.obstacle_centers[j][1])
                if x_dist <= (self.obstacle_widths[j] / 2 + self.collision_radius) and y_dist <= (self.obstacle_heights[j] / 2 + self.collision_radius):
                    collisions[i] = True
                    
        return collisions

    def check_agent_boundary_collisions(self, robot_positions):
        '''
        Checks for robot-boundary collisions.
        
        Output: Array with collision booleans (True for collision, else False).
        '''
        n = len(robot_positions)
        collisions = [False] * n
        for i in range(n):
            if robot_positions[i][0] <= self.collision_radius or robot_positions[i][0] >= self.map_size[0] - self.collision_radius:
                collisions[i] = True
            if robot_positions[i][1] <= self.collision_radius or robot_positions[i][1] >= self.map_size[1] - self.collision_radius:
                collisions[i] = True
        return collisions

    def check_collisions(self, robot_positions):
        '''
        Checks whether the robots have any kind of collisions.
        
        Output: Array with collision booleans (True for collision, else False).
        '''
        agent_collisions = self.check_agent_collisions(robot_positions)
        agent_obstacle_collisions = self.check_agent_obstacle_collisions(robot_positions)
        agent_boundary_collisions = self.check_agent_boundary_collisions(robot_positions)

        collisions = [any(x) for x in zip(agent_collisions, agent_obstacle_collisions, agent_boundary_collisions)]
        return collisions

    def plotter(self, collisions, robot_positions):
        '''
        Plotter for visual inspection.
        '''
        fig, ax = plt.subplots()
        for i, (x, y) in enumerate(robot_positions):
            if collisions[i]:
                circle = plt.Circle((x, y), self.collision_radius, color='r', alpha=0.3)
            else:
                circle = plt.Circle((x, y), self.collision_radius, color='g', alpha=0.3)
            ax.add_patch(circle)
            ax.text(x, y, f'Robot {i}', color='k', ha='center', va='center')
            ax.scatter(x, y, color='k', s=10)

        for i, (x, y) in enumerate(self.obstacle_centers):
            rect = plt.Rectangle((x - self.obstacle_widths[i] / 2, y - self.obstacle_heights[i] / 2), self.obstacle_widths[i], self.obstacle_heights[i], color='b', alpha=0.3)
            ax.add_patch(rect)

        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

if __name__ == '__main__':
    # Example usage
    map_size = (100, 100)
    num_robots = 20
    robot_positions = np.random.rand(num_robots, 2) * np.array(map_size)
    collision_radius = 5.1
    obstacle_centers = np.array([(35.1, 45), (75, 65)])
    obstacle_widths = [10, 8]
    obstacle_heights = [20, 15]
    
    obstacle_detector = CollisionDetection(map_size, collision_radius, obstacle_centers, obstacle_widths, obstacle_heights)
    collisions = obstacle_detector.check_collisions(robot_positions)
    print(collisions)
    
    obstacle_detector.plotter(collisions, robot_positions)



